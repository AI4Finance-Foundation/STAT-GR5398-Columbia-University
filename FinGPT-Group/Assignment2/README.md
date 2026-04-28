# GR5398 26 Spring: FinGPT Large Language Model Track

## Assignment 2 Instruction

### 0. Targets

In this assignment, we would like you to modify and compress your fine-tuned FinGPT model into a sentiment analysis module, which will be implemented into our Quantitative Trading pipeline. You can see what we have done in building up this pipeline in [ArXiv](https://arxiv.org/abs/2603.21330).

Specifically, you are expected to:

- Adapt your fine-tuned LLM to function as a sentiment analyst and improve its prediction accuracy
- Design a method to convert the model's text output into a numerical indicator or signal suitable for downstream quantitative use
- Improve the model's ability to predict future price trend direction

**Deliverables:** Write up your methodology and findings as a Medium blog post (following the same format as Assignment 1), and upload your code to GitHub.

**Due Date: April 20th, 2026**

---

### 1. Turning FinGPT into an Analysis Module

The core task is to make your fine-tuned FinGPT model useful as a modular input to a quantitative strategy. Below are several directions you can explore — you are encouraged to implement more than one.

#### 1.1 Price Trend Predictor

Your fine-tuned model already outputs a directional prediction on future price movement (an interval rather than a precise price). This prediction can serve directly as a trading signal. Your task is to **improve its accuracy** — consider how you might refine the model's output format, calibrate its confidence, or post-process its predictions to make them more reliable as signals.

#### 1.2 Sentiment Scorer

Beyond price predictions, the model also generates opinions and qualitative judgments about a given stock. Your task is to **convert this free-text output into a structured sentiment score** (e.g., a numerical value on a fixed scale) that can be passed into a quantitative strategy. Think carefully about how to design a robust and consistent scoring mechanism.

#### 1.3 News Analyst

Financial news is a critical driver of short-term price movement. You can fine-tune or prompt-engineer your model to **specialize in news analysis**, producing a structured output — such as an event type, sentiment label, or urgency score — that the pipeline can act on.

#### 1.4 Your Own Ideas

You are encouraged to explore additional methods beyond those listed above. If you have a novel approach to extracting useful signals from a language model for quantitative purposes, try it and document your results.

---

### 2. Report Requirements

Your Medium blog post should address the following:

- **Module design:** How you structured your FinGPT model as a reusable module within a quantitative strategy pipeline
- **Methodology:** Which methods you chose, why you chose them, and how you implemented them
- **Results:** What performance improvements or signal quality gains you observed
- *(Optional)* Additional ideas or experiments you explored beyond the required scope
