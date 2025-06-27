import streamlit as st
import pandas as pd
from itertools import combinations
import streamlit as st
from streamlit.components.v1 import html
from datetime import datetime
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Excipient Match Maker", layout="wide")

if "formulation_history" not in st.session_state:
    st.session_state.formulation_history = []

if "formulation_counter" not in st.session_state:
    st.session_state.formulation_counter = 0



# --- Inject custom CSS ---
st.markdown(
    """
    <style>
    body {
        background-color: #ffffff;
        color: #000000;
    }

    .top-banner {
        background-color: #008080;
        padding: 1rem;
        color: white;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
    }

    .tag {
        display: inline-block;
        background-color: #008080;
        color: white;
        padding: 5px 12px;
        margin: 3px 5px 3px 0;
        border-radius: 15px;
        font-weight: 500;
        font-size: 14px;
    }

    ul {
        list-style-type: none;
        padding-left: 0;
    }

    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Teal banner ---
st.markdown('<div class="top-banner">Excipient Match Maker</div>', unsafe_allow_html=True)


# Load Excipient Descriptions from Excel
desc_file = "Excipient Descriptions.xlsx"
desc_df = pd.read_excel(desc_file)

# Strip whitespace and create dictionary for lookup
desc_df.columns = desc_df.columns.str.strip()
desc_df['Excipient'] = desc_df['Excipient'].str.strip()
desc_df['Description'] = desc_df['Description'].str.strip()

excipient_descriptions = dict(zip(desc_df['Excipient'], desc_df['Description']))

explanation_file = "Excipient Incapability Explanation.xlsx"

try:
    explanation_df = pd.read_excel(explanation_file)
    explanation_df.columns = explanation_df.columns.str.strip()

    incompatibility_explanations = {}
    for _, row in explanation_df.iterrows():
        try:
            excipient1 = str(row['Excipient1']).strip()
            excipient2 = str(row['Excipient2']).strip()
            rationale = str(row['Rationale']).strip()
            pair = tuple(sorted([excipient1, excipient2]))
            incompatibility_explanations[pair] = rationale
        except KeyError as e:
            st.warning(f"Skipping row due to missing column: {e}")
except FileNotFoundError:
    st.error(f"Error: The file '{explanation_file}' was not found. Please ensure it's in the same directory as your script.")
    incompatibility_explanations = {}
except KeyError as e:
    st.error(f"Error reading '{explanation_file}': Missing expected column. Please ensure it has 'Excipient1', 'Excipient2', and 'Rationale' columns. Detail: {e}")
    incompatibility_explanations = {}


def get_hover_html(excipient):
    desc = excipient_descriptions.get(excipient, "No description available.")
    return f"""
    <style>
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin: 4px;
    }}
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 250px;
        background-color: #f9f9f9;
        color: #333;
        text-align: left;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        font-size: 13px;
    }}
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>

    <div class="tooltip">
        <span style="background-color: #e0f7fa; color: #006064; padding: 6px 12px;
                     border-radius: 16px; font-size: 13px; font-weight: 500;
                     display: inline-block; border: 1px solid #b2ebf2;">
            {excipient}
        </span>
        <div class="tooltiptext">
            <b>{excipient}</b><br>
            <hr style='margin: 5px 0;'>
            <small>{desc}</small>
        </div>
    </div>
    """

def get_incompat_hover_html(pair, severity, explanations_dict):
    excipient1, excipient2 = pair
    # Ensure the pair is sorted for dictionary lookup
    sorted_pair = tuple(sorted([excipient1, excipient2]))
    explanation = explanations_dict.get(sorted_pair, "No detailed explanation available for this specific incompatibility.")

    color = "red" if severity == "Major" else "orange"
    text_color = "white"

    return f"""
    <style>
    .incompat-tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
        margin-right: 5px; /* Adjust spacing as needed */
    }}
    .incompat-tooltip .incompat-tooltiptext {{
        visibility: hidden;
        width: 300px; /* Wider for explanations */
        background-color: #f9f9f9;
        color: #333;
        text-align: left;
        border: 1px solid #ccc;
        border-radius: 8px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%; /* Position above the text */
        left: 50%;
        transform: translateX(-50%);
        opacity: 0;
        transition: opacity 0.3s;
        box-shadow: 0px 2px 8px rgba(0,0,0,0.1);
        font-size: 13px;
    }}
    .incompat-tooltip:hover .incompat-tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    </style>
    <div class="incompat-tooltip">
        <span style="color: {color}; font-weight: bold;">
            {severity} incompatibility
        </span>
        <div class="incompat-tooltiptext">
            <b>{excipient1} & {excipient2}:</b><br>
            <hr style='margin: 5px 0;'>
            <small>{explanation}</small>
        </div>
    </div>
    """

# --- Load data ---
@st.cache_data
def load_data():
    df = pd.read_excel("Excipient Incompatibilty Grid.xlsx", index_col=0)
    df.index = df.index.str.strip()
    df.columns = df.columns.str.strip()
    excipients = sorted(set(df.index) | set(df.columns))
    return df, excipients

df, excipient_list = load_data()

# --- Incompatibility logic ---
def get_incompatibility_sets(df):
    major_incompat = set()
    minor_incompat = set()
    for row in df.index:
        for col in df.columns:
            val = df.loc[row, col]
            pair = tuple(sorted([row, col]))
            if val == 2:
                major_incompat.add(pair)
            elif val == 1:
                minor_incompat.add(pair)
    return major_incompat, minor_incompat

def check_compatibility(excipients, major_incompat, minor_incompat):
    issues = []
    for a, b in combinations(excipients, 2):
        pair = tuple(sorted([a.strip(), b.strip()]))
        if pair in major_incompat:
            issues.append((pair, "Major"))
        elif pair in minor_incompat:
            issues.append((pair, "Minor"))
    return issues

def load_formulation_from_history(formulation_id):
    """Loads a saved formulation's data into session state for display."""
    for formulation in st.session_state.formulation_history:
        if formulation["id"] == formulation_id:
            st.session_state.final_excipients = formulation["excipients"].copy()
            st.session_state.issues = check_compatibility(formulation["excipients"], major_incompat, minor_incompat)
            st.session_state.show_results = True
            break
    st.rerun() # Rerun to display the results for the loaded formulation


major_incompat, minor_incompat = get_incompatibility_sets(df)

# --- Initialize session state ---
if "show_results" not in st.session_state:
    st.session_state.show_results = False


# === INPUT SECTION (only visible if not on results page) ===
if not st.session_state.show_results:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Select excipients for your formulation:")
        selected_excipients = st.multiselect(
            "Choose from existing excipients:",
            options=excipient_list,
            key="existing_select"
        )

    with col2:
        st.subheader("Formulation:")
        if st.session_state.get("existing_select"):
            tags_html = "".join(get_hover_html(excipient) for excipient in selected_excipients)
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("No excipients selected.")

    st.markdown("")
    if st.button("Check for Incompatibilities"):
        selected = st.session_state["existing_select"]
        if selected:
            # Save to history
            st.session_state.formulation_counter += 1
            form_id = st.session_state.formulation_counter # Assign a unique ID
            new_formulation = {
                "id": form_id, # Add the ID here
                "name": f"Formulation {form_id}",
                "excipients": selected.copy(),
            }
            st.session_state.formulation_history.insert(0, new_formulation)

            st.session_state.final_excipients = selected.copy()
            st.session_state.issues = check_compatibility(selected, major_incompat, minor_incompat)
            st.session_state.show_results = True
            st.rerun()


    # --- Contact Information for Excipients not found ---
    st.markdown("---")
    st.subheader("Excipient Information & Support")
    st.markdown(
        """
        For excipients not found in the list or any related questions, please contact:
        **Ewelina Randall** ([ewelina.randall@merck.com](mailto:ewelina.randall@merck.com))
        """
    )



# === RESULTS SECTION (simulated new page) ===
else:
    left, right = st.columns([2, 1])  # wider left column for results

    with left:
        st.subheader("Incompatibility Check Result")

        if not st.session_state.final_excipients:
            st.info("No excipients selected.")
        elif not st.session_state.issues:
            st.success("✅ This formulation is COMPATIBLE.")
        else:
            st.error("❌ This formulation is INCOMPATIBLE.")
            st.markdown("#### Issues Found:")
            for (pair, severity) in st.session_state.issues:
                incompat_text_html = get_incompat_hover_html(pair, severity, incompatibility_explanations)
                st.markdown(f"- **{pair[0]}** & **{pair[1]}** → {incompat_text_html}", unsafe_allow_html=True)

    with right:
        st.subheader("Formulation Summary")
        selected = st.session_state.final_excipients
        if selected:
            selected_sorted = sorted(selected)
            hover_tags_html = "".join(get_hover_html(e) for e in selected_sorted)
            st.markdown(hover_tags_html, unsafe_allow_html=True)
        else:
            st.markdown("*No excipients selected.*")

    st.markdown("---")  

    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Nodal Graph")

        excipient_count = len(st.session_state.final_excipients)

        if excipient_count > 30:
            st.warning(
                f"Nodal graph rendering is disabled for formulations with more than 30 excipients "
                f"to avoid performance issues. Please refer to the adjacency matrix for compatibility insights."
            )
        else:
            def draw_compatibility_graph(excipients):
                G = nx.Graph()
                for excipient in excipients:
                    G.add_node(excipient)
                for a, b in combinations(excipients, 2):
                    pair = tuple(sorted([a, b]))
                    if pair in major_incompat:
                        G.add_edge(a, b, color='red', label='Major')
                    elif pair in minor_incompat:
                        G.add_edge(a, b, color='orange', label='Minor')
                    else:
                        G.add_edge(a, b, color='green', label='')

                num_nodes = len(G.nodes())
                k_value = 0.5 / (num_nodes**0.5) if num_nodes > 0 else 0.5
                pos = nx.spring_layout(G, k=k_value, iterations=50, seed=42)

                edge_colors = [G[u][v]['color'] for u, v in G.edges()]
                edge_labels = {k: v for k, v in nx.get_edge_attributes(G, 'label').items() if v}
                
                fig_width = max(10, num_nodes * 2.0)
                fig_height = max(8, num_nodes * 1.5)

                plt.figure(figsize=(fig_width, fig_height))
                node_size = 5000
                nx.draw(
                    G, pos, with_labels=False, node_color='skyblue',
                    edge_color=edge_colors, width=2, node_size=node_size
                )

                font_size = 14
                if num_nodes > 5:
                    font_size = max(10, 20 - num_nodes)

                for node, (x, y) in pos.items():
                    plt.text(
                        x, y, node, fontsize=font_size, fontweight='bold',
                        ha='center', va='center',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.4')
                    )

                edge_label_font_size = max(9, 14 - num_nodes)
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='black', font_size=edge_label_font_size)
                
                plt.axis('off')
                plt.tight_layout()
                st.pyplot(plt)

            draw_compatibility_graph(st.session_state.final_excipients)

    with col2:
        st.markdown("#### Adjacency Matrix")
        import seaborn as sns
        import numpy as np

        def plot_compatibility_matrix(excipients):
            excipients = [e.strip() for e in excipients]
            size = len(excipients)
            matrix = np.zeros((size, size))

            for i in range(size):
                for j in range(i + 1, size):
                    pair = tuple(sorted([excipients[i], excipients[j]]))
                    if pair in major_incompat:
                        matrix[i, j] = 2
                        matrix[j, i] = 2  # Mirror for symmetry
                    elif pair in minor_incompat:
                        matrix[i, j] = 1
                        matrix[j, i] = 1  # Mirror for symmetry
                    else:
                        matrix[i, j] = 0
                        matrix[j, i] = 0  # Explicitly set compatible

            mask = np.tril(np.ones_like(matrix, dtype=bool))

            cmap = sns.color_palette(["#88e388", "#FFD700", "#FF4C4C"])  # Green, Yellow, Red

            plt.figure(figsize=(7.5, 7))
            sns.heatmap(matrix,
                        mask=mask,
                        cmap=cmap,
                        square=True,
                        linewidths=0.5,
                        linecolor='gray',
                        xticklabels=excipients,
                        yticklabels=excipients,
                        cbar=False,
                        annot=False,
                        vmin=0, vmax=2)  # Ensure color mapping is consistent

            # Gray out lower triangle
            ax = plt.gca()
            for y in range(size):
                for x in range(size):
                    if mask[y, x]:
                        ax.add_patch(plt.Rectangle((x, y), 1, 1, fill=True, color='#D3D3D3', ec='gray'))

            plt.xticks(rotation=45, ha='right', fontsize=9, fontweight='bold')
            plt.yticks(rotation=0, fontsize=9, fontweight='bold')
            plt.tight_layout()
            st.pyplot(plt)

        plot_compatibility_matrix(st.session_state.final_excipients)

    st.markdown("---")
    if st.button("Create Another Formulation"):
        st.session_state.show_results = False
        st.rerun()




def rename_formulation(idx):
    key = f"rename_input_{idx}"
    new_name = st.session_state.get(key, "").strip()
    old_name_key = f"old_name_{idx}"

    # Only update if name changed and is not empty
    if new_name and new_name != st.session_state.get(old_name_key, ""):
        st.session_state.formulation_history[idx]["name"] = new_name
        st.session_state[old_name_key] = new_name

    st.session_state.renaming_index = None  # exit editing mode

with st.sidebar:
    st.markdown("## Formulation History")
    st.markdown("")

    if not st.session_state.formulation_history:
        st.markdown("No saved formulations.")
    else:
        if "renaming_index" not in st.session_state:
            st.session_state.renaming_index = None

        for idx, f in enumerate(st.session_state.formulation_history):
            col1, col2, col3 = st.columns([5, 1, 1])

            with col1:
                if st.session_state.renaming_index == idx:
                    old_name_key = f"old_name_{idx}"
                    if old_name_key not in st.session_state:
                        st.session_state[old_name_key] = f["name"]

                    new_name = st.text_input(
                        "",
                        value=f["name"],
                        key=f"rename_input_{idx}",
                        label_visibility="collapsed",
                        placeholder="Rename formulation and press Enter",
                        on_change=rename_formulation,
                        args=(idx,),
                    )
                else:
                    # Make the formulation name a clickable button
                    if st.button(f.get('name', f"Formulation {f['id']}"), key=f"load_formulation_{idx}"):
                        load_formulation_from_history(f["id"]) # Call the new function

            with col2:
                if st.button("✎", key=f"edit_{idx}"):
                    st.session_state.renaming_index = idx

            with col3:
                if st.button("✖", key=f"delete_{idx}"):
                    st.session_state.formulation_history.pop(idx)
                    if st.session_state.renaming_index == idx:
                        st.session_state.renaming_index = None
                    # Break to avoid index errors after deletion
                    break








