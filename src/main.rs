use async_trait::async_trait;
use chrono::Utc;
use dotenv::dotenv;
use futures::future::join_all;
use lazy_static::lazy_static;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::env;
use std::sync::{Arc, Mutex};
use uuid::Uuid;
use warp::Filter;

// ======== Models / Data Structures ========

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Transaction {
    id: String,
    content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Record {
    id: String,
    transaction: Transaction,
    consensus: bool,
    details: String,
    timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AIResponse {
    agent_name: String,
    is_valid: bool,
}

#[derive(Deserialize)]
struct ValidateRequest {
    statement: String,
}

#[derive(Serialize)]
struct ValidateResponse {
    record: Record,
    ai_responses: Vec<AIResponse>,
}

// In-memory ledger
lazy_static! {
    static ref LEDGER: Mutex<Vec<Record>> = Mutex::new(Vec::new());
}

// ======== Expanded LLM Models ========

const AI_MODELS: [&str; 9] = [
    "openai/gpt-4o",
    "openai/gpt-3.5-turbo",
    "anthropic/claude-3-haiku",
    "google/gemini-pro",
    "mistral/mistral-7b-instruct",
    "meta/llama-3-70b-instruct",
    "cohere/command-r-plus",
    "deepseek/deepseek-coder",
    "x/grok",
];

// ======== AI Provider Trait ========

#[async_trait]
trait AIProvider: Send + Sync {
    async fn validate(&self, text: &str) -> Result<AIResponse, Box<dyn std::error::Error + Send + Sync>>;
}

// ======== OpenRouter Provider ========

struct OpenRouterProvider {
    client: Client,
    model: String,
}

impl OpenRouterProvider {
    fn new(model: &str) -> Self {
        Self {
            client: Client::new(),
            model: model.to_string(),
        }
    }
}

#[async_trait]
impl AIProvider for OpenRouterProvider {
    async fn validate(&self, text: &str) -> Result<AIResponse, Box<dyn std::error::Error + Send + Sync>> {
        // Use the OPENROUTER_API_KEY from your environment
        let api_key = env::var("OPENROUTER_API_KEY")
            .expect("OPENROUTER_API_KEY not set");
        
        let request_body = json!({
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": format!("Is the following statement valid? Respond ONLY with 'yes' or 'no': '{}'", text)
                }
            ]
        });

        let response = self.client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .json(&request_body)
            .send()
            .await?;
        
        let response_body: serde_json::Value = response.json().await?;
        let reply = response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("no response")
            .to_lowercase();

        let is_valid = reply.contains("yes");
        Ok(AIResponse {
            agent_name: self.model.clone(),
            is_valid,
        })
    }
}

// ======== Grok Provider ========

struct GrokProvider {
    client: Client,
}

impl GrokProvider {
    fn new() -> Self {
        // Note: For testing purposes, we accept invalid certificates.
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .danger_accept_invalid_certs(true)
                .build()
                .unwrap(),
        }
    }
}

#[async_trait]
impl AIProvider for GrokProvider {
    async fn validate(&self, text: &str) -> Result<AIResponse, Box<dyn std::error::Error + Send + Sync>> {
        let api_key = env::var("X_API_KEY").unwrap_or_else(|_| {
            eprintln!("Grok API key missing or invalid. Using fallback.");
            "dummy_key".to_string()
        });

        let request_body = json!({
            "model": "grok-2-latest",
            "messages": [
                {
                    "role": "user",
                    "content": format!("Is the following statement valid? Respond ONLY with 'yes' or 'no': '{}'", text)
                }
            ],
            "temperature": 0.1
        });

        let response = self.client
            .post("https://api.x.ai/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", api_key))
            .header("X-API-Version", "2023-11-22")
            .json(&request_body)
            .send()
            .await;

        // If there is an error, log it and mark the response as invalid.
        let response = match response {
            Ok(resp) => resp,
            Err(e) => {
                eprintln!("Grok API error: {}", e);
                return Ok(AIResponse {
                    agent_name: "x/grok".to_string(),
                    is_valid: false,
                });
            }
        };

        let response_body: serde_json::Value = response.json().await?;
        let reply = response_body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("no response")
            .to_lowercase();

        let is_valid = reply.contains("yes");
        Ok(AIResponse {
            agent_name: "x/grok".to_string(),
            is_valid,
        })
    }
}

// ======== Handler Logic ========

async fn handle_validate(
    req: ValidateRequest,
    providers: Arc<Vec<Box<dyn AIProvider>>>,
) -> Result<impl warp::Reply, warp::Rejection> {
    // 1. Create a transaction
    let transaction = Transaction {
        id: Uuid::new_v4().to_string(),
        content: req.statement.clone(),
    };

    // 2. Query all AI providers in parallel
    let futures = providers.iter().map(|p| p.validate(&transaction.content));
    let responses: Vec<AIResponse> = join_all(futures)
        .await
        .into_iter()
        .filter_map(Result::ok)
        .collect();

    // 3. Build details string from responses
    let mut details = String::new();
    for r in &responses {
        let vote_str = if r.is_valid { "yes" } else { "no" };
        details.push_str(&format!("{} voted: {}\n", r.agent_name, vote_str));
    }
    details.push_str(&format!("\nRecorded at {}\n", Utc::now().to_rfc3339()));

    // 4. Create a record and add it to our ledger
    let record = Record {
        id: Uuid::new_v4().to_string(),
        transaction,
        consensus: true,
        details,
        timestamp: Utc::now().to_rfc3339(),
    };
    {
        let mut ledger = LEDGER.lock().unwrap();
        ledger.push(record.clone());
    }

    // 5. Return the response as JSON
    let response = ValidateResponse {
        record,
        ai_responses: responses,
    };
    Ok(warp::reply::json(&response))
}

fn with_providers(
    providers: Arc<Vec<Box<dyn AIProvider>>>,
) -> impl Filter<Extract = (Arc<Vec<Box<dyn AIProvider>>>,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || providers.clone())
}

// A simple GET route at `/` for basic info
fn root_handler() -> impl warp::Reply {
    warp::reply::html("<h1>Truth Terminal MVP</h1><p>POST to /api/validate with {\"statement\":\"...\"}</p>")
}

// ======== MAIN SERVER ========

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    dotenv().ok();

    // Build AI providers based on our new list.
    // Use GrokProvider if the model is "x/grok", else use OpenRouterProvider.
    let mut ai_providers: Vec<Box<dyn AIProvider>> = Vec::new();
    for model in AI_MODELS.iter() {
        if *model == "x/grok" {
            ai_providers.push(Box::new(GrokProvider::new()));
        } else {
            ai_providers.push(Box::new(OpenRouterProvider::new(model)));
        }
    }
    let providers = Arc::new(ai_providers);

    // Define routes
    let root = warp::path::end().map(root_handler);
    let validate_route = warp::path!("api" / "validate")
        .and(warp::post())
        .and(warp::body::json())
        .and(with_providers(providers.clone()))
        .and_then(handle_validate);

    // Combine routes and add CORS support
    let routes = root
        .or(validate_route)
        .with(
            warp::cors()
                .allow_any_origin()
                .allow_header("content-type")
                .allow_methods(vec![warp::http::Method::POST, warp::http::Method::GET])
        );

    println!("Server running on 0.0.0.0:3030");
    warp::serve(routes).run(([0, 0, 0, 0], 3030)).await;
    Ok(())
}







