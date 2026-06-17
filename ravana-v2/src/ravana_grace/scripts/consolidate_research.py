import os
import json
from pathlib import Path

# Discover repository root
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def consolidate_research_package(output_file=None):
    if output_file is None:
        output_file = os.path.join(root_dir, "paper", "RAVANA_v2_FINAL_PACKAGE.md")
        
    print(f"Consolidating RAVANA v2 research results into {output_file}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        # 1. Header
        f.write("# RAVANA v2: Complete Research & Validation Package\n")
        f.write(f"Generated on: 2026-03-22\n\n")
        f.write("> This document consolidates the manuscript, empirical data summaries, and raw JSON result files into a single archival format.\n\n")
        
        # 2. Append PAPER_MANUSCRIPT_DATA.md
        data_path = os.path.join(root_dir, "reports", "PAPER_MANUSCRIPT_DATA.md")
        if os.path.exists(data_path):
            f.write("## PART I: Empirical Data Summary\n\n")
            with open(data_path, "r", encoding="utf-8") as data_f:
                f.write(data_f.read())
            f.write("\n\n---\n\n")
        else:
            print(f"Warning: Could not find {data_path}")
            
        # 3. Append LaTeX Manuscript
        tex_path = os.path.join(root_dir, "paper", "RAVANA_v2_Manuscript.tex")
        if os.path.exists(tex_path):
            f.write("## PART II: LaTeX Manuscript Source\n\n")
            f.write("```latex\n")
            with open(tex_path, "r", encoding="utf-8") as tex_f:
                f.write(tex_f.read())
            f.write("\n```\n\n---\n\n")
        else:
            print(f"Warning: Could not find {tex_path}")
            
        # 4. Append JSON Results
        f.write("## PART III: Raw Validation Results (JSON)\n\n")
        
        results_dir = Path(os.path.join(root_dir, "results"))
        # Define priority result files to include
        priority_results = [
            "long_horizon/final_results.json",
            "classroom_pilot_results.json",
            "adversarial_safety_summary.json",
            "exp1_bias_resistance.json",
            "exp2_metric_ablation.json",
            "exp3_cross_domain.json"
        ]
        
        for rel_path in priority_results:
            full_path = results_dir / rel_path
            if full_path.exists():
                f.write(f"### Result File: `{rel_path}`\n\n")
                f.write("```json\n")
                with open(full_path, "r", encoding="utf-8") as json_f:
                    try:
                        data = json.load(json_f)
                        # Pretty print but limit large lists
                        if "episodes" in data: del data["episodes"]
                        if "phases" in data and len(data["phases"]) > 20: 
                            data["phases"] = data["phases"][:5] + ["... (truncated for package) ..."]
                        f.write(json.dumps(data, indent=2))
                    except Exception as e:
                        f.write(f"Error reading JSON: {str(e)}")
                f.write("\n```\n\n")
        
    print(f"Successfully created: {output_file}")

if __name__ == "__main__":
    consolidate_research_package()
