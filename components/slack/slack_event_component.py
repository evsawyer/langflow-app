import json
from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message

class SlackEventComponent(Component):
    display_name = "Slack Event Component"
    description = "Processes Slack JSON payloads for reactions and app mentions."
    documentation: str = "https://docs.langflow.org/components-custom-components"
    icon = "slack"
    name = "SlackEventComponent"

    inputs = [
        MessageTextInput(
            name="input_value",
            display_name="Input Value",
            info="JSON input containing a Slack event",
            value='{"type": "reaction_added", "user": "U08FNUF1BRD", "reaction": "neutral_face", "item": {"type": "message", "channel": "C08GBSCSEG3", "ts": "1742505236.481359"}, "item_user": "U08H0711BU0", "event_ts": "1742505257.000300"}',
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Event Output", name="event_output", method="build_output"),
    ]

    def build_output(self) -> Message:
        try:
            # Parse the input JSON string
            event_data = json.loads(self.input_value)
            files_list = event_data.get("files", []) # Safely get files list or empty list

            slack_urls = []
            if files_list: # Check if the list is not empty
                for file_data in files_list: # Corrected variable name
                    url = file_data.get("url_private_download") # Safely get the URL
                    if url: # Only append if a URL was actually found
                        slack_urls.append(url)
            # Join the URLs into a single string (will be "" if list is empty)
            urls_string = "\n".join(slack_urls)

            
            event_type = event_data.get("type", "")

            if event_type == "reaction_added":
                # Extract the reaction value
                reaction = event_data.get("reaction", "No reaction found")
                output = f"{reaction}\n{urls_string}"
            elif event_type == "app_mention":
                # Extract the message text
                message = event_data.get("text", "No message found")
                output = f"{message}\n{urls_string}"
            else:
                output = {"error": "Unknown event type"}

        except json.JSONDecodeError:
            output = {"error": "Invalid JSON input"}

        # Convert the output dictionary to a JSON string
        output_str = json.dumps(output)
        print(output_str)

        # Create a Message object with the stringified output
        message = Message(text=output_str)
        
        return message
