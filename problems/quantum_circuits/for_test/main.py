from context import build_context
from baseline import entrypoint
from validate import validate

if __name__ == "__main__":
    context = build_context()
    result = entrypoint(context)

    print(validate((context,result)))