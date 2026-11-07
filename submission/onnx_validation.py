"""Validate a data-only ONNX controller against its simulator manifest."""


def reject_external_tensors(message, tensor_type):
    if isinstance(message, tensor_type) and (
            message.external_data or message.data_location == tensor_type.EXTERNAL):
        raise ValueError("ONNX external_data tensors are not permitted")
    for descriptor, value in message.ListFields():
        if descriptor.type != descriptor.TYPE_MESSAGE:
            continue
        repeated = (descriptor.is_repeated if hasattr(descriptor, "is_repeated")
                    else descriptor.label == descriptor.LABEL_REPEATED)
        for child in value if repeated else (value,):
            reject_external_tensors(child, tensor_type)


def validate_graph(model_bytes, manifest):
    import onnx
    model = onnx.load_model_from_string(model_bytes)
    reject_external_tensors(model, onnx.TensorProto)
    onnx.checker.check_model(model, full_check=True)
    inputs = {f["name"]: (onnx.TensorProto.FLOAT, f["shape"]) for f in manifest["observations"]}
    outputs = {"action": (onnx.TensorProto.FLOAT, [12])}
    if manifest["state_dim"]:
        dtype = onnx.TensorProto.DOUBLE if manifest["state_dtype"] == "float64" else onnx.TensorProto.FLOAT
        inputs["state"] = (dtype, [manifest["state_dim"]])
        outputs["next_state"] = (dtype, [manifest["state_dim"]])
    for values, expected, label in ((model.graph.input, inputs, "input"),
                                     (model.graph.output, outputs, "output")):
        if len(values) != len(expected) or {v.name for v in values} != set(expected):
            raise ValueError(f"Controller {label} names must be {list(expected)}")
        for value in values:
            dtype, shape = expected[value.name]
            tensor = value.type.tensor_type
            dims = tensor.shape.dim
            if tensor.elem_type != dtype or len(dims) != len(shape) + 1:
                raise ValueError(f"Invalid {label} type/shape for {value.name}")
            for dim, size in zip(dims, [1, *shape]):
                if dim.HasField("dim_value") and dim.dim_value != size:
                    raise ValueError(f"Invalid {label} shape for {value.name}; expected batch + {shape}")
