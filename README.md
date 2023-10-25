# storyteller
Uncensored storyteller that works with local Llama or Mistral models as well as proprietary models (TODO)

# Requirements

Install ollama and pull your favorite model. "Mistral" latest works well.
$ pip install -r requirements.txt


# To Run
Update the PLOT in story.py
$ python story.py 

This will generate 3 files
- story-{datetime stamp}.txt
- session-{datetime stamp}.txt
- chapters-{datetime stamp}.json

Story holds the complete 10 chapter story
Session holds all the prompts sent and responses received
Chapters is the working JSON of the chapters

Because LLMs do not always create correct JSON, we try 5 times before giving up. 

The better models can do it more consistently.

# TODO
- Add OpenAI, Claude, etc. option and bindings
- Add a UI
- Explain how to install ollama or link to docs
