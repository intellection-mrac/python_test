import urwid

# Sample materials data
materials = {
    "Cement": {"price": 5.0, "unit_type": "bags", "available_quantity": 100, "selected": False},
    "Sand": {"price": 3.0, "unit_type": "kgs", "available_quantity": 200, "selected": False},
    "Gravel": {"price": 8.0, "unit_type": "kgs", "available_quantity": 150, "selected": False},
    "Bricks": {"price": 0.25, "unit_type": "pieces", "available_quantity": 500, "selected": False},
    "Steel": {"price": 12.0, "unit_type": "kg", "available_quantity": 300, "selected": False},
}

# Global search query
search_query = ""

# Function to create the UI layout
def draw_ui(selected_material_idx, search_query):
    # Create header
    header = urwid.Text("Material Cost Estimator")
    search_box = urwid.Edit(f"Search: {search_query}")

    # Filter materials based on search query
    filtered_materials = [name for name in materials if search_query.lower() in name.lower()]

    material_widgets = []
    for idx, material in enumerate(filtered_materials):
        details = materials[material]
        selected = urwid.CheckBox(f"[{'X' if details['selected'] else ' '}] {material}")
        material_widgets.append(selected)

    list_box = urwid.ListBox(urwid.SimpleFocusListWalker(material_widgets))

    # Buttons (Search, Select, Finish)
    button_widgets = [
        urwid.Button("Search", on_press=on_search),
        urwid.Button("Select", on_press=on_select),
        urwid.Button("Finish", on_press=on_finish),
    ]

    # Create the layout
    pile = urwid.Pile([header, search_box] + [urwid.Text("-" * 50)] + material_widgets + [urwid.Text("-" * 50)] + button_widgets)
    
    # We want a layout overlay on top of a solid background fill, centered and with some margins
    return urwid.Overlay(
        pile, 
        urwid.SolidFill(u"\u2588"),  # Background
        align="center", valign="middle",  # Horizontal and vertical alignment
        width=('relative', 60), 
        height=('relative', 80), 
        top=2, 
        left=2
    )

# Search button action
def on_search(button, user_data):
    global search_query
    search_query = urwid.prompt("Search term:")
    loop.draw_screen()

# Select button action
def on_select(button, user_data):
    selected_material = filtered_materials[selected_material_idx]
    materials[selected_material]["selected"] = not materials[selected_material]["selected"]
    loop.draw_screen()

# Finish button action
def on_finish(button, user_data):
    raise urwid.ExitMainLoop()

# Run the application
def main():
    global loop
    top = urwid.Overlay(
        draw_ui(0, search_query), 
        urwid.SolidFill(u"\u2588"),  # Background
        align='center', valign="middle",  # Added valign here
        width=('relative', 60), 
        height=('relative', 80), 
        top=2, 
        left=2
    )
    loop = urwid.MainLoop(top, unhandled_input=handle_input)
    loop.run()

# Input handler function
def handle_input(key):
    if key == 'esc':
        raise urwid.ExitMainLoop()
    elif key == 'backspace':
        search_query = search_query[:-1]  # Remove the last character from search query

if __name__ == "__main__":
    main()

