## Session 10: LLM Servers


| 📰 Session Sheet                                                                                                                            | ⏺️ Recording                                                                                                                                           | 🖼️ Slides                                              | 👨‍💻 Repo    | 📝 Homework                                                 | 📁 Feedback                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------- | ------------- | ----------------------------------------------------------- | --------------------------------------------------- |
| [Session 10: LLM Servers](https://github.com/AI-Maker-Space/The-AI-Engineering-Certification-v1.0/tree/main/00_Docs/Modules/10_LLM_Servers) | [Recording!](https://us02web.zoom.us/rec/share/zXd6__uO2RwCmJUmNyGKY01sbwYjjrkpDDNPbfK_Es0MANaqRpFOqqYX4sEVYY1d.gJwTZk1729siXnjj) passcode: `^1$@$R@.` | [Session 10 Slides](https://canva.link/953giejzt5igxvw) | You are here! | [Session 10 Assignment](https://forms.gle/hc1B1bkTuXzNVrZU) | [Feedback 7/2](https://forms.gle/uj2QvYjHfHKFFQ8a6) |


**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU'RE FINISHED YOUR ASSIGNMENT !!!⚠️**

# Build 🏗️

In today's assignment, we'll be creating Fireworks AI endpoints, and then building a RAG application.

- 🤝 Breakout Room #1
  - Set-up Open Source Endpoint (Instructions [here](./ENDPOINT_SETUP.md)) ((This process may take 15-20min.))
  - Test Endpoint and Embeddings with the `endpoint_slammer.ipynb` notebook.
- 🤝 Breakout Room #2
  - Use the Open Source Endpoints to build a RAG LangGraph application



# Ship 🚢

The completed notebook and your RAG app/notebook!

### Deliverables

- A short Loom of either:
  - the notebook and the RAG application you built for the Main Homework Assignment; or
  - the notebook you created for the Advanced Build



# Share 🚀

Make a social media post about your final application!

### Deliverables

- Make a post on any social media platform about what you built!

Here's a template to get you started:

```
🚀 Exciting News! 🚀

I am thrilled to announce that I have just built and shipped a RAG application powered by open-source endpoints! 🎉🤖

🔍 Three Key Takeaways:
1️⃣
2️⃣
3️⃣

Let's continue pushing the boundaries of what's possible in the world of AI and question-answering. Here's to many more innovations! 🚀
Shout out to @AIMakerspace !

#LangChain #QuestionAnswering #RetrievalAugmented #Innovation #AI #TechMilestone

Feel free to reach out if you're curious or would like to collaborate on similar projects! 🤝🔥
```



# Submitting You Homework



## Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Follow the instructions in `ENDPOINT_SETUP.md`
2. Replace both `model` values in `endpoint_slammer.ipynb` with the `gpt-oss` endpoint you created in Step 1
3. Run the code cells in `endpoint_slammer.ipynb`
4. Respond to the questions in the section below
5. Build a sample RAG
6. Record a Loom video reviewing what you have learned from this session

**⚠️!!! PLEASE BE SURE TO SHUTDOWN YOUR DEDICATED ENDPOINT ON FIREWORKS AI WHEN YOU HAVE FINISHED YOUR ASSIGNMENT !!!⚠️**

## Questions



### ❓ Question #1:

What is the difference between serverless and dedicated endpoints?

#### ✅ Answer:

The difference is whether the GPU is shared or reserved for us, and how we get charged.

In Serverless, Fireworks runs the model on shared, multi-tenant hardware. We do not have to provision anything: we just call the model ID (`accounts/fireworks/models/gpt-oss-20b`) and it works! We pay per token, the api scales with demand, and we only pay for what we use, when we use it (in other words, idle does not cost anything). On the cons side, we share capacity, so our throughput and latency are not guaranteed and can slow under platform load from other tenants. This is something to avoid if we need guarantted in terms of capacity and latency.

For Dedicated endpoints, on the other end, Fiireworks provisions GPUs that run only our model. We get charged per hour for as long as the deployment exists, whether we use it of not. In return the capacity of the GPU(s) is guaranteed, and we can expect predictable performance. 

### ❓ Question #2:

Why is it important to consider token throughput and latency when choosing an LLM for user-facing applications?

#### ✅ Answer:

For a user-facing app, how fast and how reliably the model responds is part of the product, at the same level of the quality of its responses. A smart model that responds slowly, or falls over when too many people use it at once, should be seen as a bad user-facing model.

Latency is how long one request takes. In a chat UI what users feel most is how long it takes for the first token from the answer to appear. If it takes too long, the user might wonder whether the experience is broken. Streaming helps here, but showing tokens as they get generated.

Throughput is how much data the endpoint produces and how many requests it can srve at once. It can help us predict how many users our  service can support. 

## Activity 1: RAGAS Evaluation with Cost Analysis

Use RAGAS to evaluate your open-source Fireworks AI powered RAG app against an OpenAI `gpt-4.1-mini` powered equivalent. Compare retrieval quality, answer faithfulness, and end-to-end accuracy across both providers.

Additionally, instrument both pipelines with **LangSmith** to capture token usage and cost per query. Use LangSmith's tracing and cost dashboards to compare the total cost of running each provider at scale. Include your evaluation results, cost breakdown, and analysis in your Loom video.

## Advanced Activity: Local Models

Swap out the Fireworks AI endpoints for **locally-running open-source models** using [Ollama](https://ollama.com/) or another local inference server of your choice. Run both your embedding model and your chat model locally, and rebuild the RAG pipeline on top of them.

- Compare quality and latency between the local setup and your Fireworks AI hosted endpoint.
- Reflect: what are the trade-offs of local models vs. managed endpoints in a production setting?

Include your findings and a demo in your Loom video.