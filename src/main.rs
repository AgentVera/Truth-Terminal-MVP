use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use uuid::Uuid;

lazy_static::lazy_static! {
    static ref LEDGER: Mutex<Vec<Block>> = Mutex::new(Vec::new());
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Transaction {
    id: String,
    content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct Block {
    id: String,
    transaction: Transaction,
    consensus: bool,
    details: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ConsensusResult {
    transaction_id: String,
    consensus: bool,
    details: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Option<Vec<Choice>>, // Handle cases where `choices` is missing
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: Message,
}

#[derive(Debug, Deserialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ErrorResponse {
    error: ApiError,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
    r#type: String,
    code: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Welcome to the Agent Consensus CLI!");

    loop {
        println!("Enter a transaction message (or 'exit' to quit):");
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        let input = input.trim().to_string();

        if input.to_lowercase() == "exit" {
            break;
        }

        let transaction = Transaction {
            id: Uuid::new_v4().to_string(),
            content: input.clone(),
        };

        println!("User submitted transaction: {:?}", transaction);

        let client = Client::new();
        let mut agent_responses = Vec::new();
        for agent in 1..=3 {
            let response = validate_transaction(&client, &transaction, agent).await?;
            agent_responses.push(response);
        }

        let consensus = form_consensus(&agent_responses);
        let block = validate_and_add_to_chain(&transaction, &consensus).await?;

        println!("Consensus Result: {:?}", consensus);
        println!("Block added to ledger: {:?}", block);

        println!("\nCurrent Ledger:");
        for block in LEDGER.lock().unwrap().iter() {
            println!("{:?}", block);
        }
    }

    Ok(())
}

async fn validate_transaction(
    client: &Client,
    transaction: &Transaction,
    agent_id: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    println!("Agent {} validating transaction: {:?}", agent_id, transaction);

    let prompt = format!(
        "Agent {}: Validate the following transaction: '{}'. Is it valid?",
        agent_id, transaction.content
    );

    let request_body = serde_json::json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0
    });

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let response_text = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await?
        .text()
        .await?;

    println!("Raw API response: {}", response_text);

    let response: Result<OpenAIResponse, serde_json::Error> = serde_json::from_str(&response_text);

    match response {
        Ok(parsed_response) => {
            let result_text = match parsed_response.choices {
                Some(choices) if !choices.is_empty() => {
                    let content = choices[0].message.content.clone(); // Clone the content
                    content.trim().to_string() // Create an owned String from the trimmed value
                }
                _ => "No valid choice found".to_string(),
            };
            

            let is_valid = result_text.contains("valid");
            println!("Agent {} validation result: {}", agent_id, is_valid);
            Ok(is_valid)
        }
        Err(_) => {
            let error_response: Result<ErrorResponse, _> = serde_json::from_str(&response_text);
            if let Ok(error) = error_response {
                println!("API Error: {}", error.error.message);
                Err(format!("OpenAI API error: {}", error.error.message).into())
            } else {
                println!("Unexpected response format: {}", response_text);
                Err("Unexpected OpenAI API response.".into())
            }
        }
    }
}


fn form_consensus(agent_responses: &[bool]) -> ConsensusResult {
    let valid_count = agent_responses.iter().filter(|&&res| res).count();
    let total_count = agent_responses.len();
    let consensus_reached = valid_count > total_count / 2;

    ConsensusResult {
        transaction_id: Uuid::new_v4().to_string(),
        consensus: consensus_reached,
        details: if consensus_reached {
            "Consensus reached: Transaction is valid.".to_string()
        } else {
            "Consensus failed: Transaction is invalid.".to_string()
        },
    }
}

async fn validate_and_add_to_chain(
    transaction: &Transaction,
    consensus: &ConsensusResult,
) -> Result<Block, Box<dyn std::error::Error>> {
    let client = Client::new();
    let prompt = format!(
        "Validate the block with transaction: '{}' and consensus result: '{}'. Should it be added to the chain?",
        transaction.content, consensus.consensus
    );

    let request_body = serde_json::json!({
        "model": "gpt-3.5-turbo",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 10,
        "temperature": 0.0
    });

    let api_key = std::env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY environment variable not set");

    let response_text = client
        .post("https://api.openai.com/v1/chat/completions")
        .header("Authorization", format!("Bearer {}", api_key))
        .json(&request_body)
        .send()
        .await?
        .text()
        .await?;

    println!("Raw API response: {}", response_text);

    let response: Result<OpenAIResponse, serde_json::Error> = serde_json::from_str(&response_text);

    match response {
        Ok(parsed_response) => {
            let result_text = match parsed_response.choices {
                Some(choices) if !choices.is_empty() => {
                    let content = choices[0].message.content.clone(); // Clone the content
                    content.trim().to_string() // Create an owned String from the trimmed value
                }
                _ => "No valid choice found".to_string(),
            };
            

            if result_text.contains("add") {
                let block = Block {
                    id: Uuid::new_v4().to_string(),
                    transaction: transaction.clone(),
                    consensus: consensus.consensus,
                    details: consensus.details.clone(),
                };

                LEDGER.lock().unwrap().push(block.clone());
                Ok(block)
            } else {
                Err("Block rejected by validation process.".into())
            }
        }
        Err(_) => {
            println!("Unexpected response format: {}", response_text);
            Err("Unexpected OpenAI API response.".into())
        }
    }
}
