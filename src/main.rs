use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Mutex;
use uuid::Uuid;
use chrono::Utc;
use std::io::Write;

use solana_client::rpc_client::RpcClient;

const COLOR_GREEN: &str = "\x1B[32m"; // Green text
const COLOR_RED: &str = "\x1B[31m";   // Red text
const COLOR_RESET: &str = "\x1B[0m";  // Reset to default text color


const AI_MODELS: [&str; 10] = [
    "GPT-3.5 (text-davinci-003)",
    "GPT-4 (gpt-4-turbo)",
    "Claude (Anthropic Claude-1)",
    "Claude 2 (Anthropic Claude-2)",
    "Llama 2 (Meta AI)",
    "Cohere Command R",
    "Mistral 7B",
    "BLOOM (Hugging Face)",
    "PaLM 2 (Google AI)",
    "OpenAssistant (LAION)",
];


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
    solana_block: u64, // Add this field
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
    
    clear_screen(); // Clear the screen
    print_banner(); // Print the ASCII art

    loop {
        prompt_text();

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
        for agent in 1..=5 {
            let response = validate_transaction(&client, &transaction, agent).await?;
            agent_responses.push(response);
        }

        // Always accept the block but record the votes
        let block = validate_and_add_to_chain(&transaction, agent_responses).await?;

        println!("Block added to ledger: {:?}", block);

        display_ledger();

        println!("\nWould you like to ask another question or exit? (Type 'continue' or 'exit'):");
        let mut choice = String::new();
        std::io::stdin().read_line(&mut choice)?;
        let choice = choice.trim().to_lowercase();

        if choice == "exit" {
            break;
        } else if choice != "continue" {
            println!("Invalid input. Exiting...");
            break;
        }
    }

    Ok(())
}

fn display_ledger() {
    println!("\n=== Current Ledger ===\n");
    let ledger = LEDGER.lock().unwrap();
    for (i, block) in ledger.iter().enumerate() {
        println!(
            "Block {}: {{ Assertion: '{}', Consensus: true }}\nVotes:\n{}",
            i + 1,
            block.transaction.content,
            block.details
        );
    }
    println!("=======================\n");
}
async fn validate_transaction(
    client: &Client,
    transaction: &Transaction,
    agent_id: usize,
) -> Result<bool, Box<dyn std::error::Error>> {
    let model_name = AI_MODELS[agent_id % AI_MODELS.len()];
    println!(
        "Agent {} ({}) validating transaction: {:?}\n",
        agent_id, model_name, transaction
    );

    let prompt = format!(
        "Agent {} ({}) is validating the following transaction: '{}'. Is it valid? Respond with 'yes' or 'no'.\n",
        agent_id, model_name, transaction.content
    );

    let request_body = serde_json::json!({
        "model": "gpt-3.5-turbo", // Using GPT-3.5 for simulation
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
                    let content = choices[0].message.content.clone();
                    content.trim().to_lowercase() // Normalize response to lowercase
                }
                _ => "no valid response".to_string(),
            };

            let is_valid = result_text.contains("yes");
            println!("Agent {} ({}) validation result: {}", agent_id, model_name, is_valid);
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
    agent_responses: Vec<bool>,
) -> Result<Block, Box<dyn std::error::Error>> {
    // Solana RPC client
    let rpc_client = RpcClient::new("https://api.mainnet-beta.solana.com");

    // Fetch the current block height
    let current_block = match rpc_client.get_slot() {
        Ok(block) => block,
        Err(err) => {
            eprintln!("Error fetching Solana block: {}", err);
            0 // Default block number if an error occurs
        }
    };

    // Record agent responses and their associated models
    let mut details = String::new();
    for (i, response) in agent_responses.iter().enumerate() {
        let vote = if *response {
            format!("{}yes{}", COLOR_GREEN, COLOR_RESET) // Green for yes
        } else {
            format!("{}no{}", COLOR_RED, COLOR_RESET)   // Red for no
        };
        let model_name = AI_MODELS[i % AI_MODELS.len()]; // Assign model name
        details.push_str(&format!("Agent {} ({}) voted: {}\n", i + 1, model_name, vote));
    }
    

    // Add timestamp and block height
    let timestamp = Utc::now();
    details.push_str(&format!(
        "\nThis block was added to Solana at block {} on {}.\n",
        current_block, timestamp
    ));

    // Create the block
    let block = Block {
        id: Uuid::new_v4().to_string(),
        transaction: transaction.clone(),
        consensus: true,
        details,
        solana_block: current_block,
    };

    // Add block to the ledger
    LEDGER.lock().unwrap().push(block.clone());
    Ok(block)
}

fn clear_screen() {
    print!("\x1B[2J\x1B[1;1H"); // ANSI escape code to clear the screen
    std::io::stdout().flush().unwrap(); // Flush the output to ensure it is displayed immediately
}


fn print_banner() {
    println!();
    println!();
    println!(
        r#"
  _______ _____  _    _ _______ _    _ 
 |__   __|  __ \| |  | |__   __| |  | |
    | |  | |__) | |  | |  | |  | |__| |
    | |  |  _  /| |  | |  | |  |  __  |
    | |  | | \ \| |__| |  | |  | |  | |
    |_|  |_|  \_\\____/   |_|  |_|  |_|"#
    );

    println!("\nWelcome to the TRUTH chain!\n");
    println!("\nBringing accountability to LLMs & AI\n");
    println!("We are currently testing the following models:\n");

    for (index, model) in AI_MODELS.iter().enumerate() {
        println!("Agent {}: {}", index + 1, model);
    }

    println!("\n");
}

fn prompt_text() {
    // Set text color to orange (RGB: 255, 165, 0)
    print!("\x1B[38;2;255;165;0mEnter a transaction message (or 'exit' to quit): \x1B[0m");
    std::io::stdout().flush().unwrap(); // Ensure the output is displayed immediately
}
