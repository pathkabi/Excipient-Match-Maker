# 💊 Excipient Match Maker

A Streamlit-based web application designed to streamline excipient compatibility screening during oral formulation design.

---

## 🚀 Overview

**Excipient Match Maker** allows formulation scientists to rapidly assess compatibility between selected excipients based on GPTeal-generated rationale. The tool visually flags major and minor incompatibilities, presents rich context for each interaction, and empowers iterative formulation development through saved histories and interactive visualizations.

---

## 🧠 Key Features

- ✅ **Compatibility Checker**: Select excipients and instantly view incompatibility flags (Major/Minor).
- 💬 **Contextual Rationale**: Explanations for each incompatibility generated using Merck sources via GPTeal.
- 🧩 **Excipient Descriptions**: Hover-over tooltips provide additional background on selected excipients.
- 🧠 **Graphical Insights**:
  - Nodal Graph to visualize incompatibility links.
  - Adjacency Matrix to understand pairwise relationships.
- 🕓 **Formulation History**: Save, rename, and reload previous formulations for continuous development.

---

## 🗂️ Repository Structure

```plaintext
📁 data/
├── Excipient Descriptions.xlsx
├── Excipient Incompatibility Explanation.xlsx
├── Excipient Incompatibility Grid.xlsx

📄 app.py         # Main Streamlit application
📄 README.md      # You're reading it
