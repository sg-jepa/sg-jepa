import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MENAGERIE_DIR = ROOT_DIR / "third_party" / "mujoco_menagerie"
UR5E_DIR = MENAGERIE_DIR / "universal_robots_ur5e"
ROBOTIQ_2F85_DIR = MENAGERIE_DIR / "robotiq_2f85"


def load_xml_root(path):
    return ET.parse(path).getroot()


def walk_xml(element):
    yield element
    for child in element:
        yield from walk_xml(child)


def implicit_mesh_name(mesh_element):
    return mesh_element.get("name") or Path(mesh_element.get("file")).stem


def collect_asset_names(xml_root):
    mesh_names = []
    material_names = []
    asset = xml_root.find("asset")
    if asset is None:
        return mesh_names, material_names
    for child in asset:
        if child.tag == "mesh":
            mesh_names.append(implicit_mesh_name(child))
        elif child.tag == "material" and child.get("name"):
            material_names.append(child.get("name"))
    return mesh_names, material_names


def prefix_menagerie_tree(element, prefix, mesh_names, material_names):
    prefixed = copy.deepcopy(element)
    for item in walk_xml(prefixed):
        for attr_name in ("class", "childclass"):
            attr_value = item.get(attr_name)
            if attr_value:
                item.set(attr_name, prefix + attr_value)
        name = item.get("name")
        if name:
            item.set("name", prefix + name)
        for attr_name in ("joint", "joint1", "joint2", "body1", "body2", "tendon", "site"):
            attr_value = item.get(attr_name)
            if attr_value:
                item.set(attr_name, prefix + attr_value)
        mesh_name = item.get("mesh")
        if mesh_name and mesh_name in mesh_names:
            item.set("mesh", prefix + mesh_name)
        material_name = item.get("material")
        if material_name and material_name in material_names:
            item.set("material", prefix + material_name)
    return prefixed


def prefixed_asset_child(child, asset_dir, prefix, mesh_names, material_names):
    prefixed = prefix_menagerie_tree(child, prefix, mesh_names, material_names)
    if child.tag == "mesh":
        prefixed.set("name", prefix + implicit_mesh_name(child))
    mesh_file = prefixed.get("file")
    if mesh_file:
        prefixed.set("file", str((asset_dir / mesh_file).resolve()))
    return prefixed


def append_prefixed_defaults(
    root, xml_root, prefix, mesh_names, material_names, insert_before_worldbody
):
    for default in xml_root.findall("default"):
        insert_before_worldbody(
            root,
            prefix_menagerie_tree(default, prefix, mesh_names, material_names),
        )


def append_prefixed_assets(
    root, xml_root, asset_dir, prefix, mesh_names, material_names, get_asset
):
    asset = get_asset(root)
    source_asset = xml_root.find("asset")
    if source_asset is None:
        return
    for child in source_asset:
        asset.append(
            prefixed_asset_child(
                child,
                asset_dir,
                prefix,
                mesh_names,
                material_names,
            )
        )


def find_body_recursive(body, name):
    if body.get("name") == name:
        return body
    for child in body.findall("body"):
        found = find_body_recursive(child, name)
        if found is not None:
            return found
    return None


def find_direct_site(body, name):
    for site in body.findall("site"):
        if site.get("name") == name:
            return site
    return None
