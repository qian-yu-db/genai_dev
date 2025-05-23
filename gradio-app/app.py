from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config
import gradio as gr
import os
from gradio.themes.utils import sizes
from databricks.sdk import WorkspaceClient
from datetime import datetime, timedelta, timezone
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the Databricks Workspace Client
w = WorkspaceClient()

# Ensure environment variable is set correctly
assert os.getenv('SERVING_ENDPOINT'), "SERVING_ENDPOINT must be set in app.yaml."
assert os.getenv('DATABRICKS_WAREHOUSE_ID'), "DATABRICKS_WAREHOUSE_ID must be set in app.yaml."

def sqlQuery(query: str) -> pd.DataFrame:
    cfg = Config() # Pull environment variables for auth
    with sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{os.getenv('DATABRICKS_WAREHOUSE_ID')}",
        credentials_provider=lambda: cfg.authenticate
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall_arrow().to_pandas()
        
def filter_customer_profile(df, start_time, end_time, phone_number):
    start_time = datetime.fromtimestamp(start_time)
    end_time = datetime.fromtimestamp(end_time)
    # print(f"start_time: {start_time}, end_time: {end_time}, phone_number: {phone_number}")
    df_filtered = df[(df["call_timestamp"] >= start_time) & (df["call_timestamp"] <= end_time)]
    if phone_number not in df_filtered['phone_number'].to_list():
        return "Invalid selection"
    df_filtered = df_filtered[(df_filtered["phone_number"] == phone_number)]
    return [df_filtered.loc[df_filtered['phone_number'] == phone_number, 'customer_name'].values[0], 
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'customer_tenancy'].values[0], 
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'email'].values[0],
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'policy_number'].values[0],
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'address'].values[0],
            df_filtered.loc[df_filtered['phone_number'] == phone_number, 'automobile'].values[0]]

def process_input(user_input):
    # Process the user input if needed
    return f"Write a email response to caller {user_input} base on the last conversation?"

def respond(message, history=False):
    endpoint = os.getenv('SERVING_ENDPOINT')
    if len(message.strip()) == 0:
        return "ERROR the question should not be empty"
    try:
        messages = []
        if history:
            for human, assistant in history:
                messages.append(ChatMessage(content=human, role=ChatMessageRole.USER))
                messages.append(
                    ChatMessage(content=assistant, role=ChatMessageRole.ASSISTANT)
                )
        messages.append(ChatMessage(content=message, role=ChatMessageRole.USER))
        response = w.serving_endpoints.query(
            name=endpoint,
            messages=messages,
            temperature=0.0,
            max_tokens=500,
            stream=False,
        )
    except Exception as error:
        return f"ERROR requesting endpoint {endpoint}: {error}"
    return response.choices[0].message.content

theme = gr.themes.Soft(
    text_size=sizes.text_sm,
    radius_size=sizes.radius_sm,
    spacing_size=sizes.spacing_sm,
)

df = sqlQuery("""select 
                    a.phone_number, 
                    b.call_timestamp,
                    b.sentiment,
                    concat(a.first_name, ' ', a.last_name) as customer_name, 
                    concat('Customer since ', cast(a.pol_issue_date as string)) as customer_tenancy, 
                    a.email, 
                    a.policy_number, 
                    a.address, 
                    concat(a.Model_YEAR, ' ', a.MAKE, ' ', a.MODEL) as automobile 
                    from fins_genai.customer_service.customer_policy_info a
                    join fins_genai.customer_service.call_transcript_analytics b on a.phone_number = b.phone_number
                    where b.sentiment = 'negative' or b.sentiment = 'mixed'
                    """)
df['call_timestamp'] = df['call_timestamp'].apply(lambda x: x.tz_localize(None).to_pydatetime())

with gr.Blocks() as app:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Unhappy Customer details by Caller ID")
            start_date = gr.DateTime(label="Start Date", value='2024-01-01 00:00:00')
            end_date = gr.DateTime(label="End Date", value='2024-12-31 00:00:00')
            phone_number_selector = gr.Dropdown(list(df['phone_number'].values), label="Caller ID")
            customer_name_output = gr.Textbox(label="Customer Name", placeholder="Customer name")
            tenancy_output = gr.Textbox(label="Customer Tenancy", placeholder="Customer Tenancy")
            email_output = gr.Textbox(label="Email", placeholder="Email")
            policy_number_output = gr.Textbox(label="Policy Number", placeholder="Policy Number")
            address_output = gr.Textbox(label="Address", placeholder="Address")
            auto_output = gr.Textbox(label="Automobile", placeholder="Automobile")
            @gr.on(inputs=[start_date, end_date, phone_number_selector], outputs=[customer_name_output, tenancy_output, email_output, policy_number_output, address_output, auto_output])
            def update_filter(start_date, end_date, phone_number_selector):
                return filter_customer_profile(df, start_date, end_date, phone_number_selector)
            process_btn = gr.Button("Process Current Customer")
            output_box_process_btn = gr.Textbox(label="CRM action")

        process_btn.click(
            process_input, 
            inputs=[phone_number_selector], 
            outputs=[output_box_process_btn]
        )
    
        with gr.Column(scale=3):
            gr.Markdown("# Databricks App - Customer Service CRM Assistant")
            gr.Markdown("## Instruction")
            gr.Markdown("1. Select a customer from the Caller ID dropdown list")
            gr.Markdown("2. Click on the Process Current Customer button")
            gr.Markdown("3. Click on the Submit button")
            chat_interface = gr.ChatInterface(
                respond,
                chatbot=gr.Chatbot(
                    show_label=False, container=False, show_copy_button=True, bubble_full_width=True, height=500,
                ),
                textbox=gr.Textbox(placeholder="Draft an email response ...", container=False, scale=0),
                #title="Databricks App - Customer Service Assistant",
                #description="This email assist tool help you draft email response based on customer's unhappy converstaion.<br> <li>It refer to customer's profile, intent, and content of call transcript.</li> <li>This data is synthetically generated for example, without support.<li> It is using Llama 3p1 70B model, can hallucinate and should not be used as production content.</li><br>",
                cache_examples=False,
                theme=theme,
                additional_inputs_accordion="Settings",
            )

        # Connect the processed output to the ChatInterface
        output_box_process_btn.change(
            lambda x: x,  # Identity function to pass the value unchanged
            inputs=[output_box_process_btn],
            outputs=[chat_interface.textbox]  # Access the textbox of ChatInterface
        )


if __name__ == "__main__":
    app.launch()
