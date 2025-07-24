use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use tokio;
use serde_json::{json, Value};

#[derive(Debug, Deserialize)]
struct StructuredInsight {
    topic: Option<String>,
    concept: Option<String>,
    definition: Option<String>,
    example: Option<String>,
}

#[derive(Default, Clone, Debug, PartialEq, Eq)]
struct Concept {
    definition: Option<String>,
    examples: HashSet<String>,
    related_concepts: HashSet<String>,
}

#[derive(Default, Clone, Debug)]
struct Knowledge {
    concepts: HashMap<String, Concept>,
}

impl Knowledge {
    fn add_concept(&mut self, concept: String) {
        self.concepts.entry(concept).or_default();
    }

    fn add_related_concept(&mut self, concept: &str, related: String) {
        self.concepts
            .entry(concept.to_string())
            .or_default()
            .related_concepts
            .insert(related);
    }

    fn add_definition(&mut self, concept: String, definition: String) {
        self.concepts.entry(concept).or_default().definition = Some(definition);
    }

    fn add_example(&mut self, concept: &str, example: String) {
        self.concepts
            .entry(concept.to_string())
            .or_default()
            .examples
            .insert(example);
    }
}

const LAMBDA_API_BASE: &str = "https://api.lambda.ai/v1";
const MODEL: &str = "llama3.1-8b-instruct";
const API_KEY: &str = "secret_llama3-8b_163a81aed8714aa7ad7704f1b68e5290.b4UQJpCA2sgSDGF90EmEcDIO3QTXpU3W";
const PRICE_INPUT_MILLION_TOKENS: f64 = 0.025;
const PRICE_OUTPUT_MILLION_TOKENS: f64 = 0.04;


#[tokio::main]
async fn main() -> Result<(), reqwest::Error> {

    let input = "Genomics";

    let prompt_text = format!(
        "How does {} relate to other fields of science?",
        input
    );

    let mut knowledge = Knowledge::default();

    build_documentation(&mut knowledge, prompt_text).await?;
    write_documentation_to_file(&knowledge);

    Ok(())
}

/// Sends a chat completion request to the OpenAI-compatible API and returns the full JSON response.
async fn get_chat_completion_json(api_base: &str, api_key: &str, model: &str, prompt: String) -> Result<Value, reqwest::Error> {
    let client = Client::new();

    let payload = json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert conversationalist who responds to the best of your ability."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    });

    let response = client
        .post(format!("{}/chat/completions", api_base))
        .header("Authorization", format!("Bearer {}", api_key))
        .header("Content-Type", "application/json")
        .json(&payload)
        .send().await?
        .json::<Value>().await?; // Deserialize directly to serde_json::Value

    // Safely extract desired fields
    let content = response["choices"]
        .get(0)
        .and_then(|c| c["message"].get("content"))
        .and_then(|c| c.as_str())
        .unwrap_or("")
        .to_string();

    let usage = &response["usage"];
    let prompt_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0);
    let completion_tokens = usage["completion_tokens"].as_u64().unwrap_or(0);
    let total_tokens = usage["total_tokens"].as_u64().unwrap_or(0);

    Ok(json!({
        "content": content,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
    }))
}


async fn build_documentation(
    knowledge: &mut Knowledge,
    initial_prompt: String,
) -> Result<(), reqwest::Error> {
    knowledge.add_concept("General".to_string());

    let json_results = get_chat_completion_json(LAMBDA_API_BASE, API_KEY, MODEL, initial_prompt.clone()).await?;

    let text_response_content: String = json_results["content"].to_string();
    let input_tokens_used:u64 = json_results["prompt_tokens"].as_u64().unwrap();
    let output_tokens_used:u64 = json_results["completion_tokens"].as_u64().unwrap();
    let total_tokens_used:u64 = json_results["total_tokens"].as_u64().unwrap();

    println!("Initial Summary: {}", text_response_content);

    extract_insights(&text_response_content, knowledge).await?;

    // Recursive exploration
    let mut explored: HashSet<String> = HashSet::new();
    let mut to_explore: Vec<String> = knowledge
    .concepts
    .keys()
    .filter(|c| *c != "General") // Skip "General"
    .cloned()
    .collect();

    let mut depth = 0;
    let mut total_input_tokens = 0;
    let mut total_output_tokens = 0;
    let mut sum_total_tokens = 0;

    while let Some(concept) = to_explore.pop() {
        if explored.contains(&concept) {
            continue;
        }

        explored.insert(concept.clone());

        let knowledge_depth = to_explore.len();

        println!("Explored concepts to date: {}", explored.len());
        println!("Remaining concepts to explore: {}", knowledge_depth);
        println!("Exploring related concept: {}", concept);

        if depth > 0{
            println!("total input tokens:{}", total_input_tokens);
            println!("total outpupt tokens: {}", total_output_tokens);
            println!("Final price for inference through lamba api:{}", ((total_input_tokens as f64*(PRICE_INPUT_MILLION_TOKENS/1000000.0)+total_output_tokens as f64*(PRICE_OUTPUT_MILLION_TOKENS/1000000.0))));
            println!("TOTAL TOKENS USED={}", sum_total_tokens);
            break;
        }

        let prompt_text = format!(
            "In the context of {}, how does the concept '{}' relate to other scientific disciplines or subfields? List related concepts, define them, and provide examples.",
            initial_prompt, concept
        );

        let mut json_results = get_chat_completion_json(LAMBDA_API_BASE, API_KEY, MODEL, prompt_text).await?;

        let text_response_content: String = json_results["content"].to_string();

        total_input_tokens += input_tokens_used + json_results["prompt_tokens"].as_u64().unwrap();
        total_output_tokens += output_tokens_used + json_results["completion_tokens"].as_u64().unwrap();
        sum_total_tokens += total_tokens_used + json_results["total_tokens"].as_u64().unwrap();

        println!("Summary for '{}': {}", concept, text_response_content);

        json_results = extract_insights(&text_response_content, knowledge).await?;

        total_input_tokens += input_tokens_used + json_results["prompt_tokens"].as_u64().unwrap();
        total_output_tokens += output_tokens_used + json_results["completion_tokens"].as_u64().unwrap();
        sum_total_tokens += total_tokens_used + json_results["total_tokens"].as_u64().unwrap();

        if let Some(concept_entry) = knowledge.concepts.get(&concept) {
            for related in &concept_entry.related_concepts {
                if !explored.contains(related) && !to_explore.contains(related) {
                    to_explore.push(related.clone());
                }
            }
        }

        depth += 1;
    }

    Ok(())
}

fn parse_escaped_json_list(raw_input: &str) -> String {
    // Step 1: Unescape the entire thing if it's an escaped JSON string
    let unescaped: String = serde_json::from_str(raw_input).expect("UNABLE TO UNESCAPE JSON"); // unescapes \" \n etc

    // Step 2: Extract JSON array from within the unescaped string
    let start = unescaped.find('[').unwrap();
    let end = unescaped.rfind(']').unwrap();
    let json_block = &unescaped[start..=end];

    // Step 3: Deserialize to Vec<StructuredInsight>
    json_block.to_string()
}

fn extract_json_block(text: &str) -> Option<String> {

    let json_block = parse_escaped_json_list(text);

    let start = json_block.find('[')?;
    let end = json_block.rfind(']')?;
    if start < end {
        Some(json_block[start..=end].to_string())
    } else {
        None
    }
}
async fn extract_insights(
    text: &str,
    knowledge: &mut Knowledge,
) -> Result<Value, reqwest::Error> {
    let prompt = format!(
        "Analyze the following text and return JSON single level list with these fields: topic, concept, definition, example.\n\
        Example JSON list format:\n\
        [{{ \"topic\": \"Physics\", \"concept\": \"Gravity\", \"definition\": \"A force...\", \"example\": \"An apple falling...\" }}]\n\n\
        Text: \"{}\"",
        text
    );

    let json_results = get_chat_completion_json(LAMBDA_API_BASE, API_KEY, MODEL, prompt).await?;

    let text_response_content: String = json_results["content"].to_string();
    
    println!("Raw model output:\n{}", text_response_content);

    match extract_json_block(&text_response_content)
        .and_then(|json| serde_json::from_str::<Vec<StructuredInsight>>(&json).ok())
    {
        Some(insights) => {
            for insight in insights {
                if let Some(concept) = &insight.concept {
                    knowledge.add_concept(concept.clone());

                    if let Some(topic) = &insight.topic {
                        knowledge.add_related_concept(concept, topic.clone());
                        knowledge.add_related_concept(topic, concept.clone()); // Add reverse link
                    }

                    if let Some(def) = &insight.definition {
                        knowledge.add_definition(concept.clone(), def.clone());
                    }

                    if let Some(ex) = &insight.example {
                        knowledge.add_example(concept, ex.clone());
                    }
                }
            }
        }
        None => {
            eprintln!("Failed to parse JSON array or extract it:\n{}", text_response_content);
        }
    }

    Ok(json_results)
}

fn write_documentation_to_file(knowledge: &Knowledge) {
    use std::io::Write;
    let mut file = std::fs::File::create("documentation.txt").expect("Failed to create file");

    for (concept, details) in &knowledge.concepts {
        writeln!(file, "Concept: {}", concept).unwrap();

        if let Some(def) = &details.definition {
            writeln!(file, "  Definition: {}", def).unwrap();
        }

        if !details.examples.is_empty() {
            writeln!(file, "  Examples:").unwrap();
            for example in &details.examples {
                writeln!(file, "    - {}", example).unwrap();
            }
        }

        if !details.related_concepts.is_empty() {
            writeln!(file, "  Related Concepts:").unwrap();
            for rc in &details.related_concepts {
                writeln!(file, "    - {}", rc).unwrap();
            }
        }

        writeln!(file).unwrap();
    }
}
