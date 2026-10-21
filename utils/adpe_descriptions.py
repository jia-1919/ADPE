"""Description loading and class/attribute alignment without GPU dependencies."""

import json
import os.path as osp
from collections import OrderedDict


def load_json(filename):
    with open(filename, encoding="utf-8") as stream:
        return json.load(stream)


def _is_imagenet_dataset(dataset_name):
    """Return True for ImageNet and ImageNet-style datasets.

    Some ImageNet classname files expose two synset pairs with duplicate
    display names. Using those display names directly as Python dict keys
    collapses the class dimension. The alias mapping below keeps dataset labels
    unchanged while selecting unique keys from descriptions_bag.json.
    """
    return "imagenet" in str(dataset_name).lower()


def _wordify_classname(name):
    """Normalize a dataset/JSON classname using ADPE's canonical convention."""
    return str(name).lower().replace("_", " ")


def _build_descriptor_lookup_classnames(
    dataset_name, classnames, gpt_descriptions=None, lookup_aliases=None
):
    """Build unique description-JSON keys without changing dataset labels.

    Returns:
        lookup_classnames: unique keys used to index descriptions_bag.json.
        prompt_classnames: original wordified names used in cls prompts.
        alias_records: records printed to verify any ImageNet alias mapping.
    """
    prompt_classnames = [_wordify_classname(c) for c in classnames]
    lookup_classnames = []
    alias_records = []
    seen = {}

    # Occurrence-based aliases used by the previously validated ImageNet ADPE
    # implementation:
    #   n04008634 (second displayed "missile") -> JSON["projectile"]
    #   n04355933 (first displayed "sunglasses") -> JSON["sunglass"]
    # The original class list and numeric labels are never modified.
    imagenet_occurrence_alias = {
        ("missile", 2): "projectile",
        ("sunglasses", 1): "sunglass",
    }

    use_imagenet_alias = _is_imagenet_dataset(dataset_name)
    for idx, cname in enumerate(prompt_classnames):
        seen[cname] = seen.get(cname, 0) + 1
        lookup = cname
        if use_imagenet_alias:
            lookup = imagenet_occurrence_alias.get((cname, seen[cname]), cname)
        lookup = (lookup_aliases or {}).get(idx, lookup)
        lookup_classnames.append(lookup)
        if lookup != cname:
            alias_records.append({
                "index": idx,
                "dataset_classname": cname,
                "json_lookup_key": lookup,
                "occurrence": seen[cname],
            })

    # All downstream prompt dictionaries require exactly one key per numeric
    # class. Fail here rather than silently collapsing classes and reshaping the
    # resulting text tensor incorrectly.
    duplicate_lookup_keys = sorted({
        name for name in lookup_classnames
        if lookup_classnames.count(name) > 1
    })
    if duplicate_lookup_keys:
        raise ValueError(
            "[ADPE ImageNet Alias] Duplicate class keys remain after alias mapping.\n"
            f"  Duplicate lookup keys: {duplicate_lookup_keys}\n"
            "  This would collapse classes in prompt dictionaries."
        )

    if gpt_descriptions is not None:
        missing = [
            name for name in lookup_classnames
            if name not in gpt_descriptions
        ]
        if missing:
            raise KeyError(
                "[ADPE ImageNet Alias] Some classes are missing from "
                "descriptions_bag.json.\n"
                f"  Missing lookup keys: {missing[:50]}\n"
                f"  Number missing: {len(missing)}\n"
                "  Check classnames.txt and descriptions_bag.json naming."
            )

    return lookup_classnames, prompt_classnames, alias_records


def load_gpt_descriptions(dataset_name, classnames, model_cfg, cls_templates, lookup_aliases=None):
    '''
    读取 GPT 生成的描述文本，并整理成 ADPE 使用的结构。
    期望格式：
        {'descriptor1': {class1: [desc1, desc2, ...], class2: [...]}, 'descriptor2': ...}
    返回：
        descriptor -> class -> list[prompt]
    '''
    # 读取数据集对应的描述文件
    project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
    # ImageNet distribution shifts share the original ImageNet descriptions.
    description_dataset = "ImageNet" if _is_imagenet_dataset(dataset_name) else dataset_name
    description_path = model_cfg.DESCRIPTIONS_PATH or osp.join(
        "descriptions", description_dataset, "descriptions_bag.json"
    )
    description_path = osp.expanduser(description_path)
    if not osp.isabs(description_path):
        description_path = osp.join(project_root, description_path)
    print(f"[ADPE] Loading descriptions: {description_path}")
    gpt_descriptions = load_json(description_path)
    if not isinstance(gpt_descriptions, dict) or not gpt_descriptions:
        raise ValueError("ADPE descriptions must be a nonempty class-to-attributes mapping")
    # JSON key 与数据集类名使用完全相同的 ADPE 规范化方式。
    gpt_descriptions = {
        _wordify_classname(k): v for k, v in gpt_descriptions.items()
    }
    lookup_classnames, prompt_classnames, alias_records = (
        _build_descriptor_lookup_classnames(
            dataset_name,
            classnames,
            gpt_descriptions=gpt_descriptions,
            lookup_aliases=lookup_aliases,
        )
    )

    if alias_records:
        print("[ADPE ImageNet Alias] Applied classname -> JSON lookup aliases:")
        for record in alias_records:
            print(
                f"  index={record['index']}: "
                f"{record['dataset_classname']} "
                f"occurrence={record['occurrence']} -> "
                f"{record['json_lookup_key']}"
            )

    # descriptor 维度（如 texture/pattern 等）
    descriptors = list(next(iter(gpt_descriptions.values())))
    if not descriptors or "cls" in descriptors:
        raise ValueError("Descriptions need attribute branches; 'cls' is reserved")
    if model_cfg.ATTR_ORDER:
        available = {name.casefold(): name for name in descriptors}
        selected = []
        for item in model_cfg.ATTR_ORDER.split(","):
            item = item.strip()
            if item.isdecimal() and 1 <= int(item) <= len(descriptors):
                selected.append(descriptors[int(item) - 1])
            elif item.casefold() in available:
                selected.append(available[item.casefold()])
            else:
                raise ValueError(f"Unknown attribute {item!r}; available: {descriptors}")
        if len(set(selected)) != len(selected):
            raise ValueError("ADPE.ATTR_ORDER must not contain duplicate attributes")
        descriptors = selected
    if model_cfg.ATTR_COUNT:
        if model_cfg.ATTR_COUNT > len(descriptors):
            raise ValueError("ADPE.ATTR_COUNT exceeds the selected attribute count")
        descriptors = descriptors[:model_cfg.ATTR_COUNT]
    if model_cfg.DESCRIPTION_SOURCE_DATASET:
        source_path = osp.join(project_root, "descriptions", model_cfg.DESCRIPTION_SOURCE_DATASET,
                               "descriptions_bag.json")
        source = load_json(source_path)
        source_descriptors = list(next(iter(source.values())))
        if descriptors != source_descriptors:
            raise ValueError(
                f"Checkpoint descriptor schema {source_descriptors} differs from target {descriptors}. "
                "For cross-dataset transfer, supply MODEL.ADPE.DESCRIPTIONS_PATH with target-class "
                "descriptions using exactly the source attribute names and order."
            )
    for classname in lookup_classnames:
        attributes = gpt_descriptions[classname]
        for descriptor in descriptors:
            descriptions = attributes.get(descriptor)
            if not isinstance(descriptions, list) or not descriptions or any(
                not isinstance(text, str) or not text.strip() for text in descriptions
            ):
                raise ValueError(f"Invalid descriptions for {classname!r}/{descriptor!r}")
    print(f"[ADPE] Attribute order ({len(descriptors)}): {descriptors}")
    # 为每个 descriptor 生成 “a photo of a xxx” 形式的文本 prompt
    reordered_gpt_descriptions = OrderedDict()
    for descriptor in descriptors:
        descriptor_prompts = OrderedDict()
        for lookup_name in lookup_classnames:
            descriptor_prompts[lookup_name] = [
                f'a photo of a {des}'
                for des in gpt_descriptions[lookup_name][descriptor]
            ]
        reordered_gpt_descriptions[descriptor] = descriptor_prompts

    # 添加 cls 分支：用 imagenet 模板生成 class-level prompt
    cls_temps = cls_templates
    cls_des = OrderedDict()
    for lookup_name, prompt_name in zip(lookup_classnames, prompt_classnames):
        cls_des[lookup_name] = [
            f'a photo of a {cls_temp.format(prompt_name)}'
            for cls_temp in cls_temps
        ]
    reordered_gpt_descriptions['cls'] = cls_des

    # 打印一个示例，便于检查描述是否正确加载
    first_lookup = lookup_classnames[0]
    print(
        f"\nExample description for class {first_lookup}: "
        f"\"{gpt_descriptions[first_lookup]}\"\n"
    )
    return reordered_gpt_descriptions, lookup_classnames

