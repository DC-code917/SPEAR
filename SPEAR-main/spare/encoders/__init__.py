from spare.modeling import SpearEncoder


TransformerEncoder = SpearEncoder
str2encoder = {"transformer": SpearEncoder}

__all__ = ["SpearEncoder", "TransformerEncoder", "str2encoder"]
