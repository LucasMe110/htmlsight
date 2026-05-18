from dataclasses import dataclass

TAXONOMY = [
    "button",
    "input",
    "textarea",
    "checkbox",
    "radio",
    "select",
    "link",
    "card",
    "navbar",
    "tabs",
    "modal",
    "table",
    "alert",
    "accordion",
    "image",
    "text",
    "container",
]


@dataclass(frozen=True)
class SelectorRule:
    class_name: str
    selector: str


SELECTORS = [
    SelectorRule("button", "button.btn, .btn, [role='button']"),
    SelectorRule(
        "input",
        "input[type='text'], input[type='email'], input[type='password'], "
        "input[type='search'], input[type='url'], input[type='tel'], input[type='number']",
    ),
    SelectorRule("textarea", "textarea"),
    SelectorRule("checkbox", "input[type='checkbox']"),
    SelectorRule("radio", "input[type='radio']"),
    SelectorRule("select", "select, .form-select"),
    SelectorRule("link", "a:not(.btn):not(.nav-link):not(.dropdown-item)"),
    SelectorRule("card", ".card"),
    SelectorRule("navbar", ".navbar"),
    SelectorRule("tabs", ".nav-tabs, .nav-pills"),
    SelectorRule("modal", ".modal.show, .modal-dialog"),
    SelectorRule("table", "table"),
    SelectorRule("alert", ".alert"),
    SelectorRule("accordion", ".accordion"),
    SelectorRule("image", "img, picture"),
    SelectorRule("text", "p, h1, h2, h3, h4, h5, h6, .lead"),
    SelectorRule("container", ".container, .container-fluid, .row, .col, .card-body"),
]


def class_id(class_name: str) -> int:
    return TAXONOMY.index(class_name)
