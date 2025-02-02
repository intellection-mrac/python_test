import curses
import csv
import os
from datetime import datetime

# File paths
MATERIALS_FILE = "materials.csv"
EXPORTS_DIR = "exports/"

# Default material data
DEFAULT_MATERIALS = {
    "Cement": {"price": 5.50, "unit_type": "kg", "available_quantity": 100, "required_quantity": 0},
    "Steel": {"price": 20.30, "unit_type": "kg", "available_quantity": 50, "required_quantity": 0},
    "Wood": {"price": 15.00, "unit_type": "kg", "available_quantity": 200, "required_quantity": 0},
    "Sand": {"price": 2.75, "unit_type": "kg", "available_quantity": 150, "required_quantity": 0},
}

# Ensure the exports directory exists
os.makedirs(EXPORTS_DIR, exist_ok=True)


# Load materials from CSV or use default
def load_materials():
    materials = DEFAULT_MATERIALS.copy()
    if os.path.exists(MATERIALS_FILE):
        with open(MATERIALS_FILE, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                if len(row) == 4:
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
            writer.writerow([name, details["price"], details["unit_type"], details["available_quantity"]])


# Add required quantity for selected material
def add_required_quantity(stdscr, materials, selected_row):
    material_names = list(materials.keys())
    selected_material = material_names[selected_row]
    details = materials[selected_material]

    stdscr.clear()
    stdscr.addstr(2, 0, f"Enter required quantity for {selected_material} (Available: {details['available_quantity']}):")
    stdscr.refresh()

    required_quantity = get_user_input(stdscr, 3, 0, 10)

    try:
        required_quantity = int(required_quantity)
        if 0 <= required_quantity <= details["available_quantity"]:
            details["required_quantity"] = required_quantity
            display_message(stdscr, f"Required quantity for {selected_material} updated to {required_quantity}.")
        else:
            display_message(stdscr, f"Invalid quantity! It must be between 0 and {details['available_quantity']}.")
    except ValueError:
        display_message(stdscr, "Invalid input! Please enter a valid number.")


# Add new material to the list
def add_new_material(stdscr, materials):
    stdscr.clear()
    stdscr.addstr(2, 0, "Enter new material name:")
    stdscr.refresh()
    name = get_user_input(stdscr, 3, 0, 20)

    if name in materials:
        display_message(stdscr, f"Material {name} already exists.")
        return

    stdscr.addstr(5, 0, "Enter price:")
    stdscr.refresh()
    price = get_user_input(stdscr, 6, 0, 10)
    try:
        price = float(price)
    except ValueError:
        display_message(stdscr, "Invalid price input.")
        return

    stdscr.addstr(8, 0, "Enter unit (e.g., kg, m3):")
    stdscr.refresh()
    unit = get_user_input(stdscr, 9, 0, 10)

    stdscr.addstr(11, 0, "Enter available quantity:")
    stdscr.refresh()
    available_quantity = get_user_input(stdscr, 12, 0, 10)
    try:
        available_quantity = int(available_quantity)
    except ValueError:
        display_message(stdscr, "Invalid quantity input.")
        return

    materials[name] = {
        "price": price,
        "unit_type": unit,
        "available_quantity": available_quantity,
        "required_quantity": 0,
    }

    display_message(stdscr, f"Material {name} added successfully.")


# Edit selected material's details (name, price, available quantity)
def edit_material(stdscr, materials, selected_row):
    material_names = list(materials.keys())
    selected_material = material_names[selected_row]
    details = materials[selected_material]

    stdscr.clear()
    stdscr.addstr(2, 0, f"Editing {selected_material}:")
    stdscr.addstr(3, 0, f"1. Price (current: €{details['price']})")
    stdscr.addstr(4, 0, f"2. Available Quantity (current: {details['available_quantity']})")
    stdscr.addstr(6, 0, "Choose option to edit (1 or 2):")
    stdscr.refresh()

    choice = get_user_input(stdscr, 7, 0, 1)

    if choice == "1":
        stdscr.addstr(9, 0, "Enter new price:")
        stdscr.refresh()
        new_price = get_user_input(stdscr, 10, 0, 10)
        try:
            details["price"] = float(new_price)
            display_message(stdscr, f"Price updated to €{new_price}.")
        except ValueError:
            display_message(stdscr, "Invalid price input!")
    elif choice == "2":
        stdscr.addstr(9, 0, "Enter new available quantity:")
        stdscr.refresh()
        new_quantity = get_user_input(stdscr, 10, 0, 10)
        try:
            details["available_quantity"] = int(new_quantity)
            display_message(stdscr, f"Available quantity updated to {new_quantity}.")
        except ValueError:
            display_message(stdscr, "Invalid quantity input!")
    else:
        display_message(stdscr, "Invalid choice!")


# Remove selected material from the list
def remove_material(stdscr, materials, selected_row):
    material_names = list(materials.keys())
    selected_material = material_names[selected_row]
    
    stdscr.clear()
    stdscr.addstr(2, 0, f"Are you sure you want to remove {selected_material}? (y/n):")
    stdscr.refresh()

    confirm = get_user_input(stdscr, 3, 0, 1)
    if confirm.lower() == "y":
        del materials[selected_material]
        display_message(stdscr, f"{selected_material} removed successfully.")
    else:
        display_message(stdscr, "Material removal canceled.")


# Display the material list in a simple table format
def display_materials(stdscr, materials, selected_row, search_term=""):
    stdscr.clear()

    # Header
    stdscr.addstr(0, 0, "Material List (use arrow keys to navigate, Enter to modify quantities):")
    stdscr.addstr(1, 0, "---------------------------------------------------")
    stdscr.addstr(2, 0, f"{'Material':<10} | {'Price (€)':<10} | {'Available Quantity':<15} | {'Required Quantity':<15}")
    stdscr.addstr(3, 0, "---------------------------------------------------")

    # Material rows
    row = 4
    material_names = list(materials.keys())
    
    # Apply search filter
    if search_term:
        material_names = [name for name in material_names if search_term.lower() in name.lower()]
    
    for idx, name in enumerate(material_names):
        details = materials[name]
        price_str = f"€{details['price']:.2f}"
        if idx == selected_row:
            stdscr.addstr(row, 0, f"{name:<10} | {price_str:<10} | {details['available_quantity']:<15} | {details['required_quantity']:<15}", curses.A_REVERSE)
        else:
            stdscr.addstr(row, 0, f"{name:<10} | {price_str:<10} | {details['available_quantity']:<15} | {details['required_quantity']:<15}")
        row += 1

    # Display search hint
    if search_term:
        stdscr.addstr(row + 1, 0, f"Search: {search_term} - Press ESC to clear search.")
    
    stdscr.refresh()


# Display the total cost for selected materials
def display_total_cost(stdscr, materials):
    total_cost = sum(details["price"] * details["required_quantity"] for details in materials.values())
    stdscr.addstr(0, 50, f"Total Cost: €{total_cost:.2f}", curses.A_BOLD)


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
    stdscr.addstr(20, 0, "--------------------------------------------------")
    for i, button in enumerate(buttons):
        if i == selected_button:
            stdscr.addstr(21, i * 12, f"[{button}]", curses.A_REVERSE)
        else:
            stdscr.addstr(21, i * 12, f"[{button}]")
    stdscr.refresh()


# Get user input with visible typing
def get_user_input(stdscr, y, x, length):
    curses.echo()  # Enable input display
    stdscr.addstr(y, x, "")
    stdscr.refresh()
    return stdscr.getstr(y, x, length).decode("utf-8").strip()


# Export current materials to CSV with timestamp
def export_to_csv(materials):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = os.path.join(EXPORTS_DIR, f"materials_{timestamp}.csv")
    with open(filename, "w", newline="") as file:
        writer = csv.writer(file)
        for name, details in materials.items():
            updated_quantity = details["available_quantity"] - details["required_quantity"]
            writer.writerow([name, details["price"], details["unit_type"], updated_quantity])
    return filename


# Export current materials to a text file with total cost for billing
def export_to_text(materials):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    total_cost = 0
    filename = os.path.join(EXPORTS_DIR, f"bill_{timestamp}.txt")
    with open(filename, "w") as file:
        file.write("Material Cost Bill\n")
        file.write("==============================\n")
        file.write(f"Timestamp: {timestamp}\n\n")
        file.write(f"{'Material':<20} {'Quantity':<10} {'Unit':<10} {'Price per Unit (€)':<15} {'Total Cost (€)'}\n")
        file.write("-" * 75 + "\n")

        for name, details in materials.items():
            if details["required_quantity"] > 0:
                cost = details["price"] * details["required_quantity"]
                total_cost += cost
                file.write(f"{name:<20} {details['required_quantity']:<10} {details['unit_type']:<10} €{details['price']:<13.2f} €{cost:.2f}\n")
        
        file.write("-" * 75 + "\n")
        file.write(f"Total Cost: €{total_cost:.2f}\n")
    return filename


# Process bill by updating available quantities and resetting required quantities
def process_bill(materials):
    for details in materials.values():
        details["available_quantity"] -= details["required_quantity"]
        details["required_quantity"] = 0  # Reset required quantity after billing


# Main loop
def main(stdscr):
    curses.curs_set(0)
    materials = load_materials()

    selected_row = 0
    selected_button = 0
    search_term = ""

    while True:
        display_materials(stdscr, materials, selected_row, search_term)
        display_total_cost(stdscr, materials)  # Show the total cost
        display_buttons(stdscr, selected_button)

        key = stdscr.getch()

        # Handle key presses for navigation and actions
        if key == curses.KEY_DOWN:
            selected_row = (selected_row + 1) % len(materials)
        elif key == curses.KEY_UP:
            selected_row = (selected_row - 1) % len(materials)

        # Handle button press (Enter)
        elif key == 10:  # Enter key
            if selected_button == 0:  # Select
                add_required_quantity(stdscr, materials, selected_row)
            elif selected_button == 1:  # New
                add_new_material(stdscr, materials)
            elif selected_button == 2:  # Edit
                edit_material(stdscr, materials, selected_row)
            elif selected_button == 3:  # Remove
                remove_material(stdscr, materials, selected_row)
            elif selected_button == 4:  # Search
                stdscr.clear()
                stdscr.addstr(2, 0, "Enter search term")
                stdscr.refresh()
                search_term = get_user_input(stdscr, 3, 0, 20)
            elif selected_button == 5:  # Export
                stdscr.clear()
                stdscr.addstr(2, 0, "Export as: [1] CSV [2] Bill (Press ESC to cancel)")
                stdscr.refresh()
                choice = get_user_input(stdscr, 3, 0, 1)
                if choice == "1":
                    filename = export_to_csv(materials)
                    display_message(stdscr, f"Materials exported to {filename}")
                elif choice == "2":
                    filename = export_to_text(materials)
                    display_message(stdscr, f"Billing details exported to {filename}")
                process_bill(materials)  # Update available quantities after export or bill
            elif selected_button == 6:  # Quit
                save_materials(materials)
                display_message(stdscr, "Saving data and quitting...")
                break

        # Handle left/right arrow for navigation between buttons
        elif key == curses.KEY_RIGHT and selected_button < 6:  # Allow right arrow until "Quit"
            selected_button += 1
        elif key == curses.KEY_LEFT and selected_button > 0:
            selected_button -= 1

        # Handle ESC key to clear search and return home or cancel export
        elif key == 27:  # ESC key
            if selected_button == 4:  # Search mode
                search_term = ""  # Clear search term
                display_materials(stdscr, materials, selected_row, search_term)  # Refresh materials list
            elif selected_button == 5:  # Export mode
                display_materials(stdscr, materials, selected_row, search_term)  # Cancel and return to list
            else:
                continue  # If ESC is pressed outside search or export mode, continue with normal flow

# Run the main function
if __name__ == "__main__":
    curses.wrapper(main)

