import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_stub(name, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


class _DummyLLM:
    def __init__(self, *args, **kwargs):
        pass

    def invoke(self, *args, **kwargs):
        return ""


class _Message:
    def __init__(self, content=None, **kwargs):
        self.content = content


class _BaseMessage(_Message):
    pass


# Stub langchain modules to avoid external LLM deps during tests.
_install_stub("langchain_openai", ChatOpenAI=_DummyLLM)
_install_stub("langchain_google_genai", ChatGoogleGenerativeAI=_DummyLLM)
_install_stub("langchain_anthropic", ChatAnthropic=_DummyLLM)

core = _install_stub("langchain_core")
messages = _install_stub(
    "langchain_core.messages",
    HumanMessage=_Message,
    SystemMessage=_Message,
    AIMessage=_Message,
    ToolMessage=_Message,
    BaseMessage=_BaseMessage,
    AnyMessage=_BaseMessage,
    BaseMessageChunk=_BaseMessage,
    AIMessageChunk=_BaseMessage,
    HumanMessageChunk=_BaseMessage,
    SystemMessageChunk=_BaseMessage,
    ToolMessageChunk=_BaseMessage,
    MessageLikeRepresentation=object,
    add_messages=lambda *args, **kwargs: None,
)
core.messages = messages


def _install_langgraph_stub():
    graph_stub = types.ModuleType("langgraph.graph")
    langgraph_stub = types.ModuleType("langgraph")

    class _StateGraph:
        def __init__(self, *args, **kwargs):
            pass

        def add_node(self, *args, **kwargs):
            return self

        def add_edge(self, *args, **kwargs):
            return self

        def add_conditional_edges(self, *args, **kwargs):
            return self

        def compile(self, *args, **kwargs):
            return object()

    graph_stub.END = object()
    graph_stub.StateGraph = _StateGraph

    sys.modules["langgraph"] = langgraph_stub
    sys.modules["langgraph.graph"] = graph_stub


_install_langgraph_stub()
