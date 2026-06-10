from __future__ import annotations

import argparse
import json

import gradio as gr

from src.common import add_config_arg, load_config
from src.generation.analyze_claim import analyze_claim


def make_app(config: dict) -> gr.Blocks:
    def run(claim: str, product_category: str, jurisdiction: str, evidence_text: str, use_llm: bool):
        record = analyze_claim(
            config=config,
            claim=claim,
            product_category=product_category,
            jurisdiction=jurisdiction,
            evidence_text=evidence_text,
            use_llm=use_llm,
        )
        evidence = record.get("retrieved_chunks", [])
        risk = {
            "risk_level": record.get("greenwashing_risk_level"),
            "risk_score": record.get("risk_score"),
            "risk_categories": record.get("risk_categories"),
            "reasoning": record.get("reasoning_summary"),
        }
        rewrite = {
            "responsible_rewrite": record.get("responsible_rewrite"),
            "consumer_explanation": record.get("consumer_explanation"),
            "ethics_notes": record.get("ethics_notes"),
            "citations": record.get("citations"),
        }
        audits = {
            "prompt_guard": record.get("prompt_guard"),
            "citation_audit": record.get("citation_audit"),
            "ethics_audit": record.get("ethics_audit"),
            "detector": record.get("detector"),
        }
        return (
            json.dumps(evidence, ensure_ascii=False, indent=2),
            json.dumps(risk, ensure_ascii=False, indent=2),
            json.dumps(rewrite, ensure_ascii=False, indent=2),
            json.dumps(audits, ensure_ascii=False, indent=2),
        )

    with gr.Blocks(title="Greenwashing Detector") as app:
        gr.Markdown("# Greenwashing Detector")
        gr.Markdown("Retrieval-augmented environmental marketing claim risk analysis and responsible rewriting.")
        with gr.Row():
            with gr.Column():
                claim = gr.Textbox(
                    label="Environmental marketing claim",
                    value="Our bottle is 100% eco-friendly and carbon neutral.",
                    lines=3,
                )
                product_category = gr.Textbox(label="Product category", value="packaging")
                jurisdiction = gr.Dropdown(
                    label="Target jurisdiction",
                    choices=["general", "US", "UK", "EU"],
                    value="general",
                )
                evidence_text = gr.Textbox(label="Optional product evidence", lines=4)
                use_llm = gr.Checkbox(label="Use Qwen LLM generation", value=True)
                submit = gr.Button("Analyze")
            with gr.Column():
                evidence = gr.Code(label="Retrieved guidance/evidence", language="json")
                risk = gr.Code(label="Risk analysis", language="json")
                rewrite = gr.Code(label="Responsible rewrite", language="json")
                audits = gr.Code(label="Audits", language="json")
        submit.click(
            run,
            inputs=[claim, product_category, jurisdiction, evidence_text, use_llm],
            outputs=[evidence, risk, rewrite, audits],
        )
    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    add_config_arg(parser)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    config = load_config(args.config)
    app = make_app(config)
    app.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
