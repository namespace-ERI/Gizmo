from Gizmo.tools.base_tool import BaseTool

class EchoTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="echo", 
            description="Echo the input", 
            parameters={"input": {"type": "string", "description": "The input to echo"}})

    def execute(self, **kwargs) -> str:
        return kwargs["input"]
