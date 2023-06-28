import argparse
import xml.etree.ElementTree as ET


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--response-path", type=str, required=True)
    args = parser.parse_args()
    return args


def pretty_print(element_tree: ET.ElementTree, indent: int = 0):

    print("--" * indent + element_tree.getroot().attrib["name"])
    for child in element_tree.getroot():
        pretty_print(ET.ElementTree(child), indent + 1)


if __name__ == "__main__":
    args = get_args()
    with open(args.response_path, "r") as f:
        response = f.read()

    etree = ET.ElementTree(ET.fromstring(response))
    pretty_print(etree)
