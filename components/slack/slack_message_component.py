from slack_sdk import WebClient
from langflow.custom import Component
from langflow.io import MessageTextInput, StrInput, Output
from langflow.schema import Data

class SlackMessageComponent(Component):
    display_name = "Slack Message Sender"
    description = "Send a message to a Slack channel."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "message-square"
    name = "SlackMessageSender"

    inputs = [
        StrInput(
            name="slack_token",
            display_name="Slack Bot Token",
            info="Your Slack bot token (starts with xoxb-)",
            value="",  # Leave empty for security reasons
        ),
        StrInput(
            name="channel_id",
            display_name="Channel ID",
            info="The ID of the Slack channel to send the message to (e.g., C08GBSCSEG3)",
            value="",
        ),
        MessageTextInput(
            name="message",
            display_name="Message",
            info="The message to send to the Slack channel",
            value="Hello from Langflow!",
        ),
        MessageTextInput(
            name="session_id",
            display_name="Session ID",
            info="The session ID of the chat. If empty, the current session ID parameter will be used.",
            advanced=True,
        ),
    ]

    outputs = [
        Output(display_name="Result", name="result", method="send_slack_message"),
    ]

    def send_slack_message(self) -> Data:
        # Ensure the slack_token is provided
        if not self.slack_token:
            error_message = "Error: Slack bot token is required."
            print(error_message)
            return Data(value=error_message)
        
        if not self.channel_id:
            error_message = "Error: Channel ID is required."
            print(error_message)
            return Data(value=error_message)
            
        # Initialize the Slack WebClient
        try:
            client = WebClient(token=self.slack_token)
            
            # Use the client to send a message
            response = client.chat_postMessage(
                channel=self.channel_id,
                thread_ts=self.session_id.rsplit('-', 1)[1],
                text=self.message
            )
            
            result = f"Message sent successfully: {response['message']['text']}"
            self.status = Data(value=result)
            return Data(value=result)
            
        except Exception as e:
            error_message = f"Error sending message to Slack: {str(e)}"
            print(error_message)
            return Data(value=error_message)