from langflow.custom import Component
from langflow.io import MessageTextInput, Output, MultilineInput
from langflow.schema import Data


class RuleLoaderComponent(Component):
    display_name = "Rule Loader"
    description = "Loads query rules and business logic for a specific table"
    name = "RuleLoader"
    icon = "brain"

    inputs = [
        MessageTextInput(
            name="table_name",
            display_name="Table Name",
            info="e.g., transactions, members",
            value="",
        ),
        MultilineInput(
            name="table_rules",
            display_name="Table Rules",
            info="Paste in business rules and defaults for this table",
            value="Paste your rules here...",
        ),
        MessageTextInput(
            name="table_name2",
            display_name="Table Name 2",
            info="e.g., transactions, members",
            value="",
        ),
        MultilineInput(
            name="table_rules2",
            display_name="Table Rules 2",
            info="Paste in business rules and defaults for this table",
            value="Paste your rules here...",
        ),
    ]

    outputs = [
        Output(name="rules", display_name="Table Rules", method="build_output"),
    ]

    def build_output(self) -> Data:
        rules = self.table_rules.strip() if self.table_rules else f"No rules provided for table: {self.table_name}"
        return Data(value=rules)
