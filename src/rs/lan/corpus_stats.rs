use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead};
use std::io::prelude::*;
use std::path::Path;

fn main() -> io::Result<()> {
    // The file path
    let path = "../../../data/nlp/WMT-14_en-de/train.en"; // Change this to your file path

    // Open the file
    let file = File::open(Path::new(path))?;
    let reader = io::BufReader::new(file);

    // Create a HashMap to store word frequencies
    let mut word_count: HashMap<String, u32> = HashMap::new();
    let mut _nlines: i64 = 0;
    let first_n: i64 = -1;

    // Read lines from the file
    for line in reader.lines() {
        let line = line?;
        _nlines += 1;
        if first_n > 0 && _nlines > first_n {
            break;
        }
        if _nlines % 10000 == 0 {
            eprintln!("{}", _nlines)
        }
        // Split the line into words, strip punctuation, and convert to lowercase
        // for word in line.split_whitespace() {
        for word in line.split_whitespace().map(|word| strip_punctuation(word).to_lowercase()) {
            // Filter out empty strings resulted from stripping punctuation
            if !word.is_empty() {
                *word_count.entry(word).or_insert(0) += 1;
            }
        }
    }

    // Print the word counts
    for (word, count) in &word_count {
        println!("{}: {}", word, count);
    }

    Ok(())
}

fn strip_punctuation(word: &str) -> String {
    word.chars().filter(|c| c.is_alphanumeric()).collect()
}
