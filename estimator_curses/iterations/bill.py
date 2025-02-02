import curses
import csv
import os
from datetime import datetime

# Default material data
DEFAULT_MATERIALS = {
    "Cement": {"price": 5.50, "unit_type": "kg", "available_quantity": 100, "required_quantity": 0},
    "Steel": {"price": 20.30, "unit_type": "kg", "available_quantity": 50, "required_quantity": 0},
    "Wood": {"price": 15.00, "unit_type": "kg", "available_quantity": 200, "required_quantity": 0},
    "Sand": {"price": 2.75, "unit_type": "kg", "available_quantity": 150, "required_quantity": 0},
}

# File paths
MATERIALS_FILE = "materials.csv"
EXPORTS_DIR = "exports/"
BILLS_FILE = "bills_history.txt"

# Ensure the exports directory exists
if not os.path.exists(EXPORTS_DIR):
    os.makedirs(EXPORTS_DIR)

if not os.path.exists(BILLS_FILE):
    with open(BILLS_FILE, "w") as f:
        f.write("")

# Load materials from CSV or use default
def load_materials():
    materials = DEFAULT_MATERIALS.copy()
    if os.path.exists(MATERIALS_FILE):
        with open(MATERIALS_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:  # Now only 4 columns (name, price, unit, available_quantity)
                    name, price, unit, available_quantity = row
                    try:
                        materials[name] = {
                            "price": float(price),
                            "unit_type": unit,
                            "available_quantity": int(available_quantity),
                            "required_quantity": 0,
                        }
                    except ValueError:
                        continue
    return materials

# Save materials to CSV (without the required quantity)
def save_materials(materials):
    with open(MATERIALS_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        for name, details in materials.items():
            # Only save available quantity (updated after subtracting required quantities)
            writer.writerow([name, details["price"], details["unit_type"], details["available_quantity"]])

# Add required quantity for selected material
def add_required_quantity(stdscr, materials, selected_row):
    material_names = list(materials.keys())
    selected_material = material_names[selected_row]
    details = materials[selected_material]

    stdscr.clear()
    stdscr.addstr(2, 0, f"Enter quantity for {selected_material} (Available: {details['available_quantity']}):")
    stdscr.refresh()

    required_quantity = get_user_input(stdscr, 3, 0, 10)

    try:
        required_quantity = int(required_quantity)
        if 0 <= required_quantity <= details["available_quantity"]:
            details["required_quantity"] = required_quantity
            display_message(stdscr, f"Updated {selected_material} to {required_quantity}.")
        else:
            display_message(stdscr, f"Quantity must be between 0 and {details['available_quantity']}.")
    except ValueError:
        display_message(stdscr, "Invalid input. Please enter a number.")

# Add new material to the list
def add_new_material(stdscr, materials):
    stdscr.clear()
    stdscr.addstr(2, 0, "Material name:")
    stdscr.refresh()
    name = get_user_input(stdscr, 3, 0, 20)

    if name in materials:
        display_message(stdscr, f"{name} already exists.")
        return

    stdscr.addstr(5, 0, "Price:")
    stdscr.refresh()
    price = get_user_input(stdscr, 6, 0, 10)
    try:
        price = float(price)
    except ValueError:
        display_message(stdscr, "Invalid price.")
        return

    stdscr.addstr(8, 0, "Unit (e.g., kg):")
    stdscr.refresh()
    unit = get_user_input(stdscr, 9, 0, 10)

    stdscr.addstr(11, 0, "Available quantity:")
    stdscr.refresh()
    available_quantity = get_user_input(stdscr, 12, 0, 10)
    try:
        available_quantity = int(available_quantity)
    except ValueError:
        display_message(stdscr, "Invalid quantity.")
        return

    materials[name] = {
        "price": price,
        "unit_type": unit,
        "available_quantity": available_quantity,
        "required_quantity": 0,
    }

    display_message(stdscr, f"{name} added successfully.")

# Remove selected material from the list
def remove_material(stdscr, materials, selected_row):
    material_names = list(materials.keys())
    selected_material = material_names[selected_row]
    
    stdscr.clear()
    stdscr.addstr(2, 0, f"Remove {selected_material}? (y/n):")
    stdscr.refresh()

    confirm = get_user_input(stdscr, 3, 0, 1)
    if confirm.lower() == "y":
        del materials[selected_material]
        display_message(stdscr, f"Removed {selected_material}.")
    else:
        display_message(stdscr, "Canceling removal.")

# Display the material list in a table format with a border
def display_materials(stdscr, materials, selected_row, search_term=""):
    stdscr.clear()

    # Header with color
    stdscr.attron(curses.color_pair(1))  # Header color
    stdscr.addstr(0, 0, "Material List (Use arrow keys to navigate, Enter to modify):")
    stdscr.addstr(1, 0, "-"*80)  # Longer partition line
    stdscr.addstr(2, 0, "|" + "Material".ljust(20) + "|" + "Price (€)".ljust(10) + "|" + "Available".ljust(20) + "|" + "Required".ljust(20) + "|")
    stdscr.addstr(3, 0, "-"*80)  # Longer partition line
    stdscr.attroff(curses.color_pair(1))  # Turn off header color

    # Material rows with selection highlight
    row = 4
    material_names = list(materials.keys())
    
    # Apply search filter
    if search_term:
        material_names = [name for name in material_names if search_term.lower() in name.lower()]
    
    for idx, name in enumerate(material_names):
        details = materials[name]
        price_str = f"€{details['price']:.2f}"
        if idx == selected_row:
            stdscr.attron(curses.color_pair(2))  # Selected item color
            stdscr.addstr(row, 0, f"|{name.ljust(20)}|{price_str.ljust(10)}|{str(details['available_quantity']).ljust(20)}|{str(details['required_quantity']).ljust(20)}|")
            stdscr.attroff(curses.color_pair(2))  # Turn off selected item color
        else:
            stdscr.addstr(row, 0, f"|{name.ljust(20)}|{price_str.ljust(10)}|{str(details['available_quantity']).ljust(20)}|{str(details['required_quantity']).ljust(20)}|")
        row += 1

    stdscr.addstr(row + 1, 0, "-" * 80)  # Longer partition line
    stdscr.refresh()

# Display a simple message in the center of the screen
def display_message(stdscr, message):
    stdscr.clear()
    stdscr.addstr(5, 0, message)
    stdscr.addstr(7, 0, "Press any key to continue.")
    stdscr.refresh()
    stdscr.getch()

# Display buttons at the bottom of the screen
def display_buttons(stdscr, selected_button):
    buttons = ["Select", "New", "Edit", "Remove", "Search", "Export", "Quit"]
    stdscr.addstr(20, 0, "-"*80)  # Longer partition line for buttons
    for i, button in enumerate(buttons):
        if i == selected_button:
            stdscr.attron(curses.color_pair(4))  # Button selected color
            stdscr.addstr(21, i*12, f"[{button}]")
            stdscr.attroff(curses.color_pair(4))  # Turn off button selected color
        else:
            stdscr.addstr(21, i*12, f"[{button}]")
    stdscr.refresh()

# Get user input with visible typing
def get_user_input(stdscr, y, x, length):
    curses.echo()  # Enable input display
    stdscr.addstr(y, x, "")
    stdscr.refresh()
    return stdscr.getstr(y, x, length).decode("utf-8").strip()

# Implement simple search algorithm
def handle_search(stdscr):
    stdscr.clear()
    stdscr.addstr(2, 0, "Enter search term:")
    stdscr.refresh()
    
    search_term = get_user_input(stdscr, 3, 0, 20)
    return search_term

def main(stdscr):
    curses.curs_set(0)

    # Initialize Nordic color scheme
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)   # Header color
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)   # Selected item color
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_GREEN)  # Total cost color
    curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_MAGENTA) # Button selected color
    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)  # Default text

    stdscr.bkgd(' ', curses.color_pair(5))  # Set background to dark grayish blue (#2E3440)

    materials = load_materials()

    selected_row = 0
    selected_button = 0
    search_term = ""  # Store the search term

    while True:
        display_materials(stdscr, materials, selected_row, search_term)
        display_buttons(stdscr, selected_button)

        key = stdscr.getch()

        if key == 27:  # ESC to quit
            break
        elif key == curses.KEY_RIGHT:
            selected_button = (selected_button + 1) % 7
        elif key == curses.KEY_LEFT:
            selected_button = (selected_button - 1) % 7
        elif key == curses.KEY_UP:
            selected_row = (selected_row - 1) % len(materials)
        elif key == curses.KEY_DOWN:
            selected_row = (selected_row + 1) % len(materials)
        elif key == 10:  # Enter
            # Handle selected option based on selected_button
            if selected_button == 0:  # Select
                add_required_quantity(stdscr, materials, selected_row)
            elif selected_button == 1:  # New
                add_new_material(stdscr, materials)
            elif selected_button == 2:  # Edit
                add_required_quantity(stdscr, materials, selected_row)
            elif selected_button == 3:  # Remove
                remove_material(stdscr, materials, selected_row)
            elif selected_button == 4:  # Search
                search_term = handle_search(stdscr)  # Search and get the term
            elif selected_button == 5:  # Export
                display_message(stdscr, "Export feature coming soon.")
            elif selected_button == 6:  # Quit
                break

        save_materials(materials)

if __name__ == "__main__":
    curses.wrapper(main)

