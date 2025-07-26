import os
import gradio as gr
from mcp.protocol import MCPMessage
from agents.coordinator_agent import CoordinatorAgent

# Directory where uploaded files will be saved
UPLOAD_DIR = "./uploaded_files"


# Initialize the CoordinatorAgent to handle ingestion, retrieval, and deletion
coordinator = CoordinatorAgent()
os.makedirs(UPLOAD_DIR, exist_ok=True)

def list_local_files():
    """
    Returns a sorted list of all files currently stored in the upload directory.
    """
    return sorted(os.listdir(UPLOAD_DIR))


def get_indexed_file_path(original_path: str, upload_dir: str = UPLOAD_DIR) -> str:
    """
    Generates a unique indexed file path to avoid name collisions.
    Example: sample.pdf → sample_1.pdf, sample_2.pdf, etc.

    Args:
        original_path (str): Original path of the uploaded file.
        upload_dir (str): Directory where the file will be saved.

    Returns:
        str: New indexed file path.
    """

    base_name = os.path.basename(original_path)
    name, ext = os.path.splitext(base_name)

    counter = 1
    while True:
        new_name = f"{name}_{counter}{ext}"
        new_path = os.path.join(upload_dir, new_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

def upload_and_ingest(files):
    """
    Handles file upload, indexing, local saving, and triggers ingestion via the coordinator.

    Args:
        files (List[str]): List of file paths selected by the user.

    Returns:
        Tuple[str, gr.update, gr.update]: 
            - String response from coordinator
            - Updated list of uploaded files for dropdown
            - Cleared file input
    """
    if not files:
        return "No files selected", gr.update(choices=list_local_files(), value=[]), gr.update(value=None)

    saved_paths = []
    for f in files:
        dst_path = get_indexed_file_path(f)
        with open(f, "rb") as src, open(dst_path, "wb") as dst:
            dst.write(src.read())
        saved_paths.append(dst_path)

    msg = MCPMessage(
        sender="UI",
        receiver="CoordinatorAgent",
        type="INGESTION_REQUEST",
        trace_id="upload_" + str(hash(saved_paths[0])),
        payload={"file_paths": saved_paths}
    )
    response = coordinator.handle_message(msg)

    return str(response), gr.update(choices=list_local_files(), value=[]), gr.update(value=None)

def query_files(user_query):
    """
    Sends a query to the LLM via the coordinator.

    Args:
        user_query (str): The user-entered question or query.

    Returns:
        str: Final response from the LLM.
    """
    if not user_query.strip():
        return "Please enter a query.", ""

    msg = MCPMessage(
        sender="UI",
        receiver="CoordinatorAgent",
        type="RETRIEVAL_REQUEST",
        trace_id="query_" + str(hash(user_query)),
        payload={"user_query": user_query}
    )
    response = coordinator.handle_message(msg)

    if response and response["type"] == "LLM_RESPONSE":
        return response["payload"]["response"], response["payload"]["citation"]
    else:
        return "Error: No valid response received", "Unknown"

def delete_files(filenames):
    """
    Deletes files from local storage and notifies the coordinator to remove it from the database.

    Args:
        filenames (str): List of the files to be deleted.

    Returns:
        Tuple[str, gr.update]: 
            - Deletion Confirmation
            - Updated list of files
    """
    if not filenames:
        return "No files selected", gr.update(choices=list_local_files(), value=[])

    deleted_msgs = []
    for filename in filenames:
        path = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(path):
            os.remove(path)

        msg = MCPMessage(
            sender="UI",
            receiver="CoordinatorAgent",
            type="DELETEFROM_DB",
            trace_id="delete_" + str(hash(filename)),
            payload={"source_file": filename}
        )
        response = coordinator.handle_message(msg)
        deleted_msgs.append(str(response))

    return f"Deleted {' '.join(filenames)}", gr.update(choices=list_local_files(), value=[])

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# Agent-Based Document Q&A System")

    with gr.Row():
        file_input = gr.File(file_types=[".pdf", ".docx", ".txt"], file_count="multiple", label="Upload Files")
        upload_btn = gr.Button("Upload & Ingest")
    upload_result = gr.Textbox(label="Ingestion Output")

    gr.Markdown("### Ask a Question")
    with gr.Row():
        query_input = gr.Textbox(label="Your Question")
        query_btn = gr.Button("Get Answer")
    answer_output = gr.Textbox(label="Answer")
    citation_output = gr.Textbox(label="Citation(s)")

    gr.Markdown("### Manage Uploaded Files")
    file_list = gr.CheckboxGroup(choices=list_local_files(), label="Uploaded Files", interactive=True)
    delete_btn = gr.Button("Delete Selected File(s)")
    delete_result = gr.Textbox(label="Deletion Output")

    # Bind actions
    upload_btn.click(fn=upload_and_ingest, inputs=file_input, outputs=[upload_result, file_list, file_input])
    query_btn.click(fn=query_files, inputs=query_input, outputs=[answer_output, citation_output])
    delete_btn.click(fn=delete_files, inputs=file_list, outputs=[delete_result, file_list])

demo.launch()
