# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parallel Streamlit Demo for PaperVizAgent
Accepts user text input, duplicates it 10 times, and runs parallel processing
to generate multiple diagram candidates for comparison.
"""

import streamlit as st
import asyncio
import base64
import json
from io import BytesIO
from PIL import Image
from pathlib import Path
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("DEBUG: Importing agents...")
import yaml
import shutil
configs_dir = Path(__file__).parent / "configs"
config_path = configs_dir / "model_config.yaml"
template_path = configs_dir / "model_config.template.yaml"

if not config_path.exists() and template_path.exists():
    print(f"DEBUG: {config_path.name} not found. Auto-generating from template")
    shutil.copy2(template_path, config_path)
try:
    from agents.planner_agent import PlannerAgent
    print("DEBUG: Imported PlannerAgent")
    from agents.visualizer_agent import VisualizerAgent
    from agents.stylist_agent import StylistAgent
    from agents.critic_agent import CriticAgent
    from agents.retriever_agent import RetrieverAgent
    from agents.vanilla_agent import VanillaAgent
    from agents.polish_agent import PolishAgent
    print("DEBUG: Imported all agents")
    from utils import config
    from utils.paperviz_processor import PaperVizProcessor
    print("DEBUG: Imported utils")

    model_config_data = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            model_config_data = yaml.safe_load(f) or {}

    def get_config_val(section, key, env_var, default=""):
        val = os.getenv(env_var)
        if not val and section in model_config_data:
            val = model_config_data[section].get(key)
        return val or default

except ImportError as e:
    print(f"DEBUG: ImportError: {e}")
    import traceback
    traceback.print_exc()
    raise e
except Exception as e:
    print(f"DEBUG: Exception during import: {e}")
    import traceback
    traceback.print_exc()
    raise e

st.set_page_config(
    layout="wide",
    page_title="PaperVizAgent Parallel Demo",
    page_icon="🍌"
)

def clean_text(text):
    """Clean text by removing invalid UTF-8 surrogate characters."""
    if not text:
        return text
    if isinstance(text, str):
        # Remove surrogate characters that cause UnicodeEncodeError
        return text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
    return text

def base64_to_image(b64_str):
    """Convert base64 string to PIL Image."""
    if not b64_str:
        return None
    try:
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
        image_data = base64.b64decode(b64_str)
        return Image.open(BytesIO(image_data))
    except Exception:
        return None

def create_sample_inputs(method_content, caption, diagram_type="Pipeline", aspect_ratio="16:9", num_copies=10, max_critic_rounds=3):
    """Create multiple copies of the input data for parallel processing."""
    base_input = {
        "filename": "demo_input",
        "caption": caption,
        "content": method_content,
        "visual_intent": caption,
        "additional_info": {
            "rounded_ratio": aspect_ratio
        },
        "max_critic_rounds": max_critic_rounds  # Add critic rounds control
    }
    
    # Create num_copies identical inputs, each with a unique identifier
    inputs = []
    for i in range(num_copies):
        input_copy = base_input.copy()
        input_copy["filename"] = f"demo_input_candidate_{i}"
        input_copy["candidate_id"] = i
        inputs.append(input_copy)
    
    return inputs

async def process_parallel_candidates(data_list, exp_mode="dev_planner_critic", retrieval_setting="auto", model_name=""):
    """Process multiple candidates in parallel using PaperVizProcessor."""
    # Create experiment config
    exp_config = config.ExpConfig(
        dataset_name="Demo",
        split_name="demo",
        exp_mode=exp_mode,
        retrieval_setting=retrieval_setting,
        model_name=model_name,
        work_dir=Path(__file__).parent,
    )
    
    # Initialize processor with all agents
    processor = PaperVizProcessor(
        exp_config=exp_config,
        vanilla_agent=VanillaAgent(exp_config=exp_config),
        planner_agent=PlannerAgent(exp_config=exp_config),
        visualizer_agent=VisualizerAgent(exp_config=exp_config),
        stylist_agent=StylistAgent(exp_config=exp_config),
        critic_agent=CriticAgent(exp_config=exp_config),
        retriever_agent=RetrieverAgent(exp_config=exp_config),
        polish_agent=PolishAgent(exp_config=exp_config),
    )
    
    # Process all candidates in parallel (concurrency controlled by processor)
    results = []
    concurrent_num = 10  # Process all 10 in parallel
    
    async for result_data in processor.process_queries_batch(
        data_list, max_concurrent=concurrent_num, do_eval=False
    ):
        results.append(result_data)
    
    return results

async def refine_image_with_nanoviz(image_bytes, edit_prompt, aspect_ratio="21:9", image_size="2K"):
    """
    Refine an image using an Image Editing API.
    
    Args:
        image_bytes: Image data in bytes
        edit_prompt: Text description of desired changes
        aspect_ratio: Output aspect ratio (21:9, 16:9, 3:2)
        image_size: Output resolution (2K or 4K)
    
    Returns:
        Tuple of (edited_image_bytes, success_message)
    """
    try:
        from google import genai
        from google.genai import types
        
        # Initialize client
        project_id = get_config_val("google_cloud", "project_id", "GOOGLE_CLOUD_PROJECT", "")
        location = get_config_val("google_cloud", "location", "GOOGLE_CLOUD_LOCATION", "global")
        
        client = genai.Client(vertexai=True, project=project_id, location=location)
        
        # Prepare content
        contents = [
            types.Part.from_text(text=edit_prompt),
            types.Part.from_bytes(
                mime_type="image/jpeg",
                data=image_bytes
            )
        ]
        
        # Configure generation
        config = types.GenerateContentConfig(
            temperature=1.0,
            max_output_tokens=8192,
            response_modalities=["IMAGE"],
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=image_size,
            ),
        )
        
        # Generate refined image
        image_model = get_config_val("defaults", "image_model_name", "IMAGE_MODEL_NAME", "")
        response = await asyncio.to_thread(
            client.models.generate_content,
            model=image_model,
            contents=contents,
            config=config
        )
        
        # Extract image from response
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    edited_image_data = part.inline_data.data
                    
                    if isinstance(edited_image_data, bytes):
                        return edited_image_data, "✅ Image refined successfully!"
                    elif isinstance(edited_image_data, str):
                        return base64.b64decode(edited_image_data), "✅ Image refined successfully!"
        
        return None, "❌ No image data found in response"
    
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


def get_evolution_stages(result, exp_mode):
    """Extract all evolution stages (images and descriptions) from the result."""
    task_name = "diagram"
    stages = []
    
    # Stage 1: Planner output
    planner_img_key = f"target_{task_name}_desc0_base64_jpg"
    planner_desc_key = f"target_{task_name}_desc0"
    if planner_img_key in result and result[planner_img_key]:
        stages.append({
            "name": "📋 Planner",
            "image_key": planner_img_key,
            "desc_key": planner_desc_key,
            "description": "Initial diagram plan based on method content"
        })
    
    # Stage 2: Stylist output (only for demo_full)
    if exp_mode == "demo_full":
        stylist_img_key = f"target_{task_name}_stylist_desc0_base64_jpg"
        stylist_desc_key = f"target_{task_name}_stylist_desc0"
        if stylist_img_key in result and result[stylist_img_key]:
            stages.append({
                "name": "✨ Stylist",
                "image_key": stylist_img_key,
                "desc_key": stylist_desc_key,
                "description": "Stylistically refined description"
            })
    
    # Stage 3+: Critic iterations
    for round_idx in range(4):  # Check up to 4 rounds
        critic_img_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        critic_desc_key = f"target_{task_name}_critic_desc{round_idx}"
        critic_sugg_key = f"target_{task_name}_critic_suggestions{round_idx}"
        
        if critic_img_key in result and result[critic_img_key]:
            stages.append({
                "name": f"🔍 Critic Round {round_idx}",
                "image_key": critic_img_key,
                "desc_key": critic_desc_key,
                "suggestions_key": critic_sugg_key,
                "description": f"Refined after critic feedback (iteration {round_idx})"
            })
    
    return stages

def display_candidate_result(result, candidate_id, exp_mode):
    """Display a single candidate result."""
    task_name = "diagram"
    
    # Determine which image to show based on exp_mode
    # For demo modes, always try to find the last critic round
    final_image_key = None
    final_desc_key = None
    
    # Try to find the last critic round
    for round_idx in range(3, -1, -1):  # Check rounds 3, 2, 1, 0
        image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
        if image_key in result and result[image_key]:
            final_image_key = image_key
            final_desc_key = f"target_{task_name}_critic_desc{round_idx}"
            break
    
    # Fallback if no critic rounds completed
    if not final_image_key:
        if exp_mode == "demo_full":
            # demo_full uses stylist before visualizer
            final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_stylist_desc0"
        else:
            # demo_planner_critic uses planner output
            final_image_key = f"target_{task_name}_desc0_base64_jpg"
            final_desc_key = f"target_{task_name}_desc0"
    
    # Display the final image
    if final_image_key and final_image_key in result:
        img = base64_to_image(result[final_image_key])
        if img:
            st.image(img, use_container_width=True, caption=f"Candidate {candidate_id} (Final)")
            
            # Add download button
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            st.download_button(
                label="⬇️ 다운로드 (Download)",
                data=buffered.getvalue(),
                file_name=f"candidate_{candidate_id}.png",
                mime="image/png",
                key=f"download_candidate_{candidate_id}",
                use_container_width=True
            )
        else:
            st.error(f"Failed to decode image for Candidate {candidate_id}")
    else:
        st.warning(f"No image generated for Candidate {candidate_id}")
    
    # Show evolution timeline in an expander
    stages = get_evolution_stages(result, exp_mode)
    if len(stages) > 1:
        with st.expander(f"🔄 진화 타임라인 보기 / View Evolution Timeline (단계: {len(stages)})", expanded=False):
            st.caption("파이프라인 단계를 거치며 다이어그램이 어떻게 진화했는지 확인하세요 (See how the diagram evolved through different pipeline stages)")
            
            for idx, stage in enumerate(stages):
                st.markdown(f"### {stage['name']}")
                st.caption(stage['description'])
                
                # Display the image for this stage
                stage_img = base64_to_image(result.get(stage['image_key']))
                if stage_img:
                    st.image(stage_img, use_container_width=True)
                
                # Show description
                if stage['desc_key'] in result:
                    with st.expander(f"📝 설명 (Description)", expanded=False):
                        cleaned_desc = clean_text(result[stage['desc_key']])
                        st.write(cleaned_desc)
                
                # Show critic suggestions if available
                if 'suggestions_key' in stage and stage['suggestions_key'] in result:
                    suggestions = result[stage['suggestions_key']]
                    with st.expander(f"💡 비평가 제안 (Critic Suggestions)", expanded=False):
                        cleaned_sugg = clean_text(suggestions)
                        if cleaned_sugg.strip() == "No changes needed.":
                            st.success("✅ 변경 필요 없음 - 반복(iteration)이 중지되었습니다. (No changes needed - iteration stopped.)")
                        else:
                            st.write(cleaned_sugg)
                
                # Add separator between stages (except for the last one)
                if idx < len(stages) - 1:
                    st.divider()
    else:
        # If only one stage, show description in simpler expander
        with st.expander(f"📝 설명 보기 (View Description)", expanded=False):
            if final_desc_key and final_desc_key in result:
                # Clean the text to remove invalid UTF-8 characters
                cleaned_desc = clean_text(result[final_desc_key])
                st.write(cleaned_desc)
            else:
                st.info("No description available")

def main():
    st.title("🍌 아티클 도식 시각화 엔진 (PaperVizAgent Demo)")
    st.markdown("AI 기반 과학 다이어그램 생성 및 다듬기 (AI-powered scientific diagram generation and refinement)")
    
    # Create tabs
    tab1, tab2 = st.tabs(["📊 처음부터 새로 생성 (Generate from Scratch)", "✨ 후보 이미지/스케치 다듬기 (Refine Uploaded Image)"])
    
    # ==================== TAB 1: Generate Candidates ====================
    with tab1:
        st.markdown("### 스케치가 없거나, 더 좋은 구도를 여러 개 받아보고 싶을 때 사용합니다. 메서드 섹션과 캡션에서 4~8개의 다이어그램 후보를 생성합니다. (Generate from scratch when no sketch is available)")
        
        # Sidebar configuration for Tab 1
        with st.sidebar:
            st.title("⚙️ 생성 설정 (Generation Settings)")
            
            exp_mode = st.selectbox(
                "파이프라인 모드 (Pipeline Mode)",
                ["demo_planner_critic", "demo_full"],
                index=0,
                key="tab1_exp_mode",
                help="사용할 에이전트 파이프라인을 선택하세요 (Select which agent pipeline to use)"
            )
            
            mode_info = {
                "demo_planner_critic": "Planner → Visualizer → Critic → Visualizer",
                "demo_full": "Retriever → Planner → Stylist → Visualizer → Critic → Visualizer. (스타일리스트는 다이어그램을 더 미적으로 만족스럽게 만들 수 있지만 과도하게 단순화되는 경향이 있습니다. 따라서 두 모드를 모두 사용해 보고 가장 좋은 모드를 선택하는 것이 좋습니다.)"
            }
            st.info(f"**파이프라인 (Pipeline):** {mode_info[exp_mode]}")
            
            retrieval_setting = st.selectbox(
                "검색 설정 (Retrieval Setting)",
                ["auto", "manual", "random", "none"],
                index=0,
                key="tab1_retrieval_setting",
                help="참조 다이어그램을 검색하는 방법: auto (자동 선택), manual (지정된 참조 사용), random (무작위 선택), none (검색 없음)"
            )
            
            num_candidates = st.number_input(
                "후보 수 (Number of Candidates)",
                min_value=4,
                max_value=8,
                value=4,
                key="tab1_num_candidates",
                help="병렬로 생성할 후보 수 (권장 4~8개) (How many parallel candidates to generate)"
            )
            
            aspect_ratio = st.selectbox(
                "화면 비율 (Aspect Ratio)",
                ["21:9", "16:9", "3:2"],
                key="tab1_aspect_ratio",
                help="생성된 다이어그램의 화면 비율 (Aspect ratio for the generated diagrams)"
            )
            
            max_critic_rounds = st.number_input(
                "최대 비평 라운드 (Max Critic Rounds)",
                min_value=1,
                max_value=5,
                value=3,
                key="tab1_max_critic_rounds",
                help="최대 비평가 개선 반복 횟수 (Maximum number of critic refinement iterations)"
            )
            
            default_model = get_config_val("defaults", "model_name", "MODEL_NAME", "YOUR_MODEL_NAME_HERE")
            options = ["", default_model] if default_model else ["", "YOUR_MODEL_NAME_HERE"]
            
            model_name = st.selectbox(
                "모델 이름 (Model Name)",
                options,
                index=0,
                key="tab1_model_name",
                help="추론에 사용할 모델 이름 (Model name to use for reasoning)"
            )
        
        st.divider()
        
        # Input section
        st.markdown("## 📝 입력 (Input)")
        
        # Example content
        example_method = r"""## Methodology: The PaperVizAgent Framework
        
        In this section, we present the architecture of PaperVizAgent, a reference-driven agentic framework for automated academic illustration. As illustrated in Figure \ref{fig:methodology_diagram}, PaperVizAgent orchestrates a collaborative team of five specialized agents—Retriever, Planner, Stylist, Visualizer, and Critic—to transform raw scientific content into publication-quality diagrams and plots. (See Appendix \ref{app_sec:agent_prompts} for prompts)

### Retriever Agent

Given the source context $S$ and the communicative intent $C$, the Retriever Agent identifies $N$ most relevant examples $\mathcal{E} = \{E_n\}_{n=1}^{N} \subset \mathcal{R}$ from the fixed reference set $\mathcal{R}$ to guide the downstream agents. As defined in Section \ref{sec:task_formulation}, each example $E_i \in \mathcal{R}$ is a triplet $(S_i, C_i, I_i)$.
To leverage the reasoning capabilities of VLMs, we adopt a generative retrieval approach where the VLM performs selection over candidate metadata:
$$
\mathcal{E} = \text{VLM}_{\text{Ret}} \left( S, C, \{ (S_i, C_i) \}_{E_i \in \mathcal{R}} \right)
$$
Specifically, the VLM is instructed to rank candidates by matching both research domain (e.g., Agent & Reasoning) and diagram type (e.g., pipeline, architecture), with visual structure being prioritized over topic similarity. By explicitly reasoned selection of reference illustrations $I_i$ whose corresponding contexts $(S_i, C_i)$ best match the current requirements, the Retriever provides a concrete foundation for both structural logic and visual style.

### Planner Agent

The Planner Agent serves as the cognitive core of the system. It takes the source context $S$, communicative intent $C$, and retrieved examples $\mathcal{E}$ as inputs. By performing in-context learning from the demonstrations in $\mathcal{E}$, the Planner translates the unstructured or structured data in $S$ into a comprehensive and detailed textual description $P$ of the target illustration:
$$
P = \text{VLM}_{\text{plan}}(S, C, \{ (S_i, C_i, I_i) \}_{E_i \in \mathcal{E}})
$$

### Stylist Agent

To ensure the output adheres to the aesthetic standards of modern academic manuscripts, the Stylist Agent acts as a design consultant.
A primary challenge lies in defining a comprehensive “academic style,” as manual definitions are often incomplete.
To address this, the Stylist traverses the entire reference collection $\mathcal{R}$ to automatically synthesize an *Aesthetic Guideline* $\mathcal{G}$ covering key dimensions such as color palette, shapes and containers, lines and arrows, layout and composition, and typography and icons (see Appendix \ref{app_sec:auto_summarized_style_guide} for the summarized guideline and implementation details). Armed with this guideline, the Stylist refines each initial description $P$ into a stylistically optimized version $P^*$:
$$
P^* = \text{VLM}_{\text{style}}(P, \mathcal{G})
$$
This ensures that the final illustration is not only accurate but also visually professional.

### Visualizer Agent

After receiving the stylistically optimized description $P^*$, the Visualizer Agent collaborates with the Critic Agent to render academic illustrations and iteratively refine their quality. The Visualizer Agent leverages an image generation model to transform textual descriptions into visual output. In each iteration $t$, given a description $P_t$, the Visualizer generates:
$$
I_t = \text{Image-Gen}(P_t)
$$
where the initial description $P_0$ is set to $P^*$.

### Critic Agent

The Critic Agent forms a closed-loop refinement mechanism with the Visualizer by closely examining the generated image $I_t$ and providing refined description $P_{t+1}$ to the Visualizer. Upon receiving the generated image $I_t$ at iteration $t$, the Critic inspects it against the original source context $(S, C)$ to identify factual misalignments, visual glitches, or areas for improvement. It then provides targeted feedback and produces a refined description $P_{t+1}$ that addresses the identified issues:
$$
P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)
$$
This revised description is then fed back to the Visualizer for regeneration. The Visualizer-Critic loop iterates for $T=3$ rounds, with the final output being $I = I_T$. This iterative refinement process ensures that the final illustration meets the high standards required for academic dissemination.

### Extension to Statistical Plots

The framework extends to statistical plots by adjusting the Visualizer and Critic agents. For numerical precision, the Visualizer converts the description $P_t$ into executable Python Matplotlib code: $I_t = \text{VLM}_{\text{code}}(P_t)$. The Critic evaluates the rendered plot and generates a refined description $P_{t+1}$ addressing inaccuracies or imperfections: $P_{t+1} = \text{VLM}_{\text{critic}}(I_t, S, C, P_t)$. The same $T=3$ round iterative refinement process applies. While we prioritize this code-based approach for accuracy, we also explore direct image generation in Section \ref{sec:discussion}. See Appendix \ref{app_sec:plot_agent_prompt} for adjusted prompts."""

        example_caption = "Figure 1: Overview of our PaperVizAgent framework. Given the source context and communicative intent, we first apply a Linear Planning Phase to retrieve relevant reference examples and synthesize a stylistically optimized description. We then use an Iterative Refinement Loop (consisting of Visualizer and Critic agents) to transform the description into visual output and conduct multi-round refinements to produce the final academic illustration."
        
        col_input1, col_input2 = st.columns([3, 2])
        
        with col_input1:
            # Example selector for method content
            method_example = st.selectbox(
                "예제 불러오기 - 메서드 (Load Example - Method)",
                ["None", "PaperVizAgent Framework"],
                key="method_example_selector"
            )
            
            # Set value based on example selection or session state
            if method_example == "PaperVizAgent Framework":
                method_value = example_method
            else:
                method_value = st.session_state.get("method_content", "")
            
            method_content = st.text_area(
                "메서드 섹션 내용 - 마크다운 권장 (Method Section Content)",
                value=method_value,
                height=250,
                placeholder="여기에 메서드 섹션 내용을 붙여넣으세요... (Paste the method section content here...)",
                help="접근 방식을 설명하는 논문의 방법론 섹션입니다. 마크다운 형식을 권장합니다."
            )
        
        with col_input2:
            # Example selector for caption
            caption_example = st.selectbox(
                "예제 불러오기 - 캡션 (Load Example - Caption)",
                ["None", "PaperVizAgent Framework"],
                key="caption_example_selector"
            )
            
            # Set value based on example selection or session state
            if caption_example == "PaperVizAgent Framework":
                caption_value = example_caption
            else:
                caption_value = st.session_state.get("caption", "")
            
            caption = st.text_area(
                "그림 캡션 - 마크다운 권장 (Figure Caption)",
                value=caption_value,
                height=250,
                placeholder="그림 캡션을 입력하세요... (Enter the figure caption...)",
                help="생성할 그림의 캡션 또는 설명입니다. 마크다운 형식을 권장합니다."
            )
        
        # Process button
        if st.button("🚀 후보 생성 (Generate Candidates)", type="primary", use_container_width=True):
            if not method_content or not caption:
                st.error("메서드 내용과 캡션을 모두 제공해주세요! (Please provide both method content and caption!)")
            else:
                # Save to session state
                st.session_state["method_content"] = method_content
                st.session_state["caption"] = caption
                
                with st.spinner(f"병렬로 {num_candidates}개의 후보를 생성하고 있습니다... 몇 분 정도 걸릴 수 있습니다. (Generating {num_candidates} candidates in parallel...)"):
                    # Create input data list
                    input_data_list = create_sample_inputs(
                        method_content=method_content,
                        caption=caption,
                        aspect_ratio=aspect_ratio,
                        num_copies=num_candidates,
                        max_critic_rounds=max_critic_rounds
                    )
                    
                    # Process in parallel
                    try:
                        results = asyncio.run(process_parallel_candidates(
                            input_data_list, 
                            exp_mode=exp_mode, 
                            retrieval_setting=retrieval_setting,
                            model_name=model_name
                        ))
                        st.session_state["results"] = results
                        st.session_state["exp_mode"] = exp_mode
                        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        st.session_state["timestamp"] = timestamp_str
                        
                        # Save results to JSON file
                        try:
                            # Create results directory if it doesn't exist
                            results_dir = Path(__file__).parent / "results" / "demo"
                            results_dir.mkdir(parents=True, exist_ok=True)
                            
                            # Generate filename with timestamp
                            json_filename = results_dir / f"demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            
                            # Save to JSON with proper encoding handling (like main.py)
                            with open(json_filename, "w", encoding="utf-8", errors="surrogateescape") as f:
                                json_string = json.dumps(results, ensure_ascii=False, indent=4)
                                # Clean invalid UTF-8 characters
                                json_string = json_string.encode("utf-8", "ignore").decode("utf-8")
                                f.write(json_string)
                            
                            st.session_state["json_file"] = str(json_filename)
                            st.success(f"✅ {len(results)}개의 후보를 성공적으로 생성했습니다! (Successfully generated {len(results)} candidates!)")
                            st.info(f"💾 결과가 다음 경로에 저장되었습니다: `{json_filename.name}`")
                        except Exception as e:
                            st.warning(f"⚠️ {len(results)}개의 후보를 생성했지만 JSON을 저장하지 못했습니다(failed to save JSON): {e}")
                    except Exception as e:
                        st.error(f"Error during processing: {e}")
                        import traceback
                        st.code(traceback.format_exc())
        
        # Display results
        if "results" in st.session_state and st.session_state["results"]:
            results = st.session_state["results"]
            current_mode = st.session_state.get("exp_mode", exp_mode)
            timestamp = st.session_state.get("timestamp", "N/A")
            
            st.divider()
            st.markdown("## 🎨 생성된 후보 (Generated Candidates)")
            st.caption(f"생성 시간: {timestamp} | 파이프라인: {mode_info.get(current_mode, current_mode)}")
            
            # Show JSON file download if available
            if "json_file" in st.session_state:
                json_file_path = Path(st.session_state["json_file"])
                if json_file_path.exists():
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.info(f"📄 결과 저장됨 (Results saved to): `{json_file_path.relative_to(Path.cwd())}`")
                    with col2:
                        with open(json_file_path, "r", encoding="utf-8") as f:
                            json_data = f.read()
                        st.download_button(
                            label="⬇️ JSON 다운로드 (Download JSON)",
                            data=json_data,
                            file_name=json_file_path.name,
                            mime="application/json",
                            use_container_width=True
                        )
            
            # Display results in a grid (3 columns)
            num_cols = 3
            num_results = len(results)
            
            for row_start in range(0, num_results, num_cols):
                cols = st.columns(num_cols)
                for col_idx in range(num_cols):
                    result_idx = row_start + col_idx
                    if result_idx < num_results:
                        with cols[col_idx]:
                            display_candidate_result(results[result_idx], result_idx, current_mode)
            
            # Add ZIP download button
            st.divider()
            st.markdown("### 💾 일괄 다운로드 (Batch Download)")
            
            try:
                import zipfile
                
                zip_buffer = BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    task_name = "diagram"
                    
                    for candidate_id, result in enumerate(results):
                        
                        # Find the final image key (same logic as display)
                        final_image_key = None
                        
                        # Try to find the last critic round
                        for round_idx in range(3, -1, -1):
                            image_key = f"target_{task_name}_critic_desc{round_idx}_base64_jpg"
                            if image_key in result and result[image_key]:
                                final_image_key = image_key
                                break
                        
                        # Fallback if no critic rounds completed
                        if not final_image_key:
                            if current_mode == "demo_full":
                                final_image_key = f"target_{task_name}_stylist_desc0_base64_jpg"
                            else:
                                final_image_key = f"target_{task_name}_desc0_base64_jpg"
                        
                        if final_image_key and final_image_key in result:
                            img = base64_to_image(result[final_image_key])
                            if img:
                                img_buffer = BytesIO()
                                img.save(img_buffer, format="PNG")
                                zip_file.writestr(
                                    f"candidate_{candidate_id}.png",
                                    img_buffer.getvalue()
                                )
                
                zip_buffer.seek(0)
                st.download_button(
                    label="⬇️ ZIP 다운로드 (Download ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name=f"papervizagent_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )
                st.success("ZIP 파일을 다운로드할 준비가 되었습니다! (ZIP file ready for download!)")
            except Exception as e:
                st.error(f"Failed to create ZIP: {e}")
    
    # ==================== TAB 2: Refine Image ====================
    with tab2:
        st.markdown("### 후보 이미지나 임의의 다이어그램을 업로드하고 레이아웃만 유지한 채 에디토리얼 스타일로 정리합니다. (Apply editorial style while keeping layout)")
        st.caption("러프한 스케치나 초안 다이어그램을 업로드하고, 원하는 변경 사항을 구체적으로 설명하여 고해상도 에디토리얼 스타일 다이어그램을 생성하세요. (Upload a sketch/diagram and refine it into an editorial illustration)")
        
        # Sidebar for refinement settings
        with st.sidebar:
            st.title("✨ 다듬기 설정 (Refinement Settings)")
            
            refine_resolution = st.selectbox(
                "대상 해상도 (Target Resolution)",
                ["2K", "4K"],
                index=0,
                key="refine_resolution",
                help="해상도가 높을수록 더 오래 걸리지만 더 나은 품질을 생성합니다. (Higher resolution takes longer but produces better quality)"
            )
            
            refine_aspect_ratio = st.selectbox(
                "화면 비율 (Aspect Ratio)",
                ["21:9", "16:9", "3:2"],
                index=0,
                key="refine_aspect_ratio",
                help="다듬어진 이미지의 화면 비율 (Aspect ratio for the refined image)"
            )
        
        st.divider()
        
        # Upload section
        st.markdown("## 📤 이미지 업로드 (Upload Image)")
        uploaded_file = st.file_uploader(
            "이미지 파일을 선택하세요 (Choose an image file)",
            type=["png", "jpg", "jpeg"],
            help="다듬고 싶은 다이어그램을 업로드하세요 (Upload the diagram you want to refine)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            uploaded_image = Image.open(uploaded_file)
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 원본 이미지 (Original Image)")
                st.image(uploaded_image, use_container_width=True)
            
            with col2:
                st.markdown("### 편집 지침 (Edit Instructions)")
                edit_prompt = st.text_area(
                    "원하는 변경 사항을 설명하세요 (Describe the changes you want)",
                    height=200,
                    placeholder="예: '레이아웃은 그대로 유지하고 색상과 선을 전문적인 에디토리얼 스타일로 세련되게 변경해줘' 또는 '텍스트를 더 크고 명확하게 만들어줘'",
                    help="수정할 내용을 설명하거나 단순히 업스케일링만 원한다면 'Keep everything the same'을 사용하세요.",
                    key="edit_prompt"
                )
                
                if st.button("✨ 이미지 다듬기 (Refine Image)", type="primary", use_container_width=True):
                    if not edit_prompt:
                        st.error("편집 지침을 제공해주세요! (Please provide edit instructions!)")
                    else:
                        with st.spinner(f"이미지를 {refine_resolution} 해상도로 다듬는 중입니다... 1분 정도 걸릴 수 있습니다."):
                            try:
                                # Convert PIL image to bytes
                                img_byte_arr = BytesIO()
                                uploaded_image.save(img_byte_arr, format='JPEG')
                                image_bytes = img_byte_arr.getvalue()
                                
                                # Call nanoviz API
                                refined_bytes, message = asyncio.run(
                                    refine_image_with_nanoviz(
                                        image_bytes=image_bytes,
                                        edit_prompt=edit_prompt,
                                        aspect_ratio=refine_aspect_ratio,
                                        image_size=refine_resolution
                                    )
                                )
                                
                                if refined_bytes:
                                    st.session_state["refined_image"] = refined_bytes
                                    st.session_state["refine_timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    st.success(message)
                                    st.rerun()
                                else:
                                    st.error(message)
                            except Exception as e:
                                st.error(f"Error during refinement: {e}")
                                import traceback
                                st.code(traceback.format_exc())
            
            # Display refined result if available
            if "refined_image" in st.session_state:
                st.divider()
                st.markdown("## 🎨 다듬은 결과 (Refined Result)")
                st.caption(f"생성 시간: {st.session_state.get('refine_timestamp', 'N/A')} | 해상도: {refine_resolution}")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 이전 (Before)")
                    st.image(uploaded_image, use_container_width=True)
                
                with col2:
                    st.markdown(f"### 이후 (After) ({refine_resolution})")
                    refined_image = Image.open(BytesIO(st.session_state["refined_image"]))
                    st.image(refined_image, use_container_width=True)
                    
                    # Download button
                    st.download_button(
                        label=f"⬇️ {refine_resolution} 이미지 다운로드 (Download Image)",
                        data=st.session_state["refined_image"],
                        file_name=f"refined_{refine_resolution}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                        mime="image/png",
                        use_container_width=True
                    )

if __name__ == "__main__":
    main()
