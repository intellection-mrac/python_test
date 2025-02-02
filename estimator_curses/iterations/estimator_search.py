import curses
import csv
import os
from datetime import datetime

# Folder to save the CSV files
CSV_FOLDER = "material_csv_files"

# Ensure the CSV folder exists
os.makedirs(CSV_FOLDER, exist_ok=True)

# Default materials list in case no CSV is found
DEFAULT_MATERIALS = {
    "Cement": {
        "price": 5.0,  # Price per unit in Euro
        "unit_type": "bags",
        "available_quantity": 100,
    },
    "Sand": {
        "price": 3.0,  # Price per unit in Euro
        "unit_type": "kgs",
        "available_quantity": 200,
    },
    "Gravel": {
        "price": 8.0,  # Price per unit in Euro
        "unit_type": "kgs",
        "available_quantity": 150,
    },
    "Bricks": {
        "price": 0.25,  # Price per unit in Euro
        "unit_type": "piece",
        "available_quantity": 500,
    },
    "Steel": {
        "price": 12.0,  # Price per unit in Euro
        "unit_type": "kg",
        "available_quantity": 300,
    },
}

def get_most_recent_csv():
    """Get the most recent CSV file based on the timestamp in the filename."""
    files = [f for f in os.listdir(CSV_FOLDER) if f.endswith(".csv")]
    if not files:
        return None
    # Sort files by creation timestamp (newest first)
    files.sort(key=lambda x: os.path.getmtime(os.path.join(CSV_FOLDER, x)), reverse=True)
    return files[0]

def load_materials_from_csv():
    """Load materials from the most recent CSV file or default materials if no CSV is found."""
    recent_csv = get_most_recent_csv()
    if not recent_csv:
        return DEFAULT_MATERIALS.copy()  # Use the default materials if no CSV found

    materials = {}
    try:
        with open(os.path.join(CSV_FOLDER, recent_csv), mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            for row in reader:
                material, price, unit_type, available_quantity = row
                materials[material] = {
                    "price": float(price),
                    "unit_type": unit_type,
                    "available_quantity": int(available_quantity)
                }
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return DEFAULT_MATERIALS.copy()  # Use default materials if there's an error reading the CSV
    return materials

def save_materials_to_csv(materials):
    """Save materials to a CSV file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"materials_{timestamp}.csv"
    filepath = os.path.join(CSV_FOLDER, filename)

    try:
        with open(filepath, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Material", "Price per Unit (€)", "Unit Type", "Available Quantity"])
            for material, details in materials.items():
                writer.writerow([material, details["price"], details["unit_type"], details["available_quantity"]])
    except Exception as e:
        print(f"Error saving CSV: {e}")

def setup_colors():
    """Initialize color pairs for the TUI interface."""
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Blue header
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # White background
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Highlighted row
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_GREEN)  # Success
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_WHITE)  # Error
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_CYAN)  # Button color

def get_quantity_input(stdscr, material_name, material_price):
    """Prompt user to input quantity for a selected material."""
    curses.echo()  # Enable echoing of input

    # Ask for quantity
    stdscr.clear()
    stdscr.addstr(2, 0, f"Enter the quantity for {material_name} (price per unit: €{material_price}):")
    try:
        quantity = int(stdscr.getstr(3, 0).decode("utf-8").strip())
        if quantity <= 0:
            return 0
        return quantity
    except ValueError:
        return 0

def draw_table(stdscr, materials, selected_materials, selected_material, selected_button, total_cost, search_query):
    """Display the list of available materials and real-time total cost."""
    stdscr.clear()

    max_width = curses.COLS - 1  # Full screen width
    stdscr.addstr(3, 0, "-" * max_width)  # Line separating header from the table

    # Search bar
    stdscr.addstr(4, 0, "Search: ", curses.A_BOLD)
    stdscr.addstr(4, 8, search_query, curses.A_BOLD | curses.color_pair(7))

    # Table columns
    stdscr.addstr(6, 0, f"{'Material':<30}{'Price per Unit (€)':<20}{'Unit Type':<15}{'Available Quantity':<20}{'Selected Quantity':<20}", curses.A_BOLD)
    stdscr.addstr(7, 0, "-" * max_width)  # Line separating header from content

    # Material list rows
    row = 8
    for idx, (material, details) in enumerate(materials.items()):
        price = details["price"]
        unit_type = details["unit_type"]
        available_quantity = details["available_quantity"]
        
        selected = "[x]" if material in selected_materials else "[ ]"
        quantity = selected_materials.get(material, 0)
        
        match = search_query.lower() in material.lower()

        # Highlight matched materials
        if match:
            stdscr.attron(curses.A_BOLD)  # Make search term bold
            stdscr.addstr(row, 0, f"{selected} {material.capitalize():<30} €{price:<20.2f} {unit_type:<15} {available_quantity:<20} {quantity:<20}")
            stdscr.attroff(curses.A_BOLD)
        else:
            stdscr.addstr(row, 0, f"{selected} {material.capitalize():<30} €{price:<20.2f} {unit_type:<15} {available_quantity:<20} {quantity:<20}")
        row += 1

    # Real-time total cost display
    stdscr.addstr(row + 1, 0, "-" * max_width)
    stdscr.addstr(row + 2, 0, f"Total Estimated Cost: €{total_cost:.2f}", curses.A_BOLD | curses.color_pair(4))

    # Button-like navigation options (Horizontal buttons)
    buttons = [
        ("Select", "Select/Deselect material"),
        ("Remove", "Remove selected material"),
        ("Finish", "Finish and view total cost"),
        ("Export", "Export remaining materials to CSV"),
        ("Add", "Add new material to inventory"),
        ("Exit", "Exit the program")  # Exit button
    ]

    button_row = row + 4
    for i, (button_label, _) in enumerate(buttons):
        button_text = f"[ {button_label} ]"
        if i == selected_button:  # Highlight the selected button
            stdscr.attron(curses.color_pair(7))
            stdscr.addstr(button_row, 0 + i * 18, button_text)
            stdscr.attroff(curses.color_pair(7))
        else:
            stdscr.addstr(button_row, 0 + i * 18, button_text)

    stdscr.refresh()

def display_summary(stdscr, selected_materials, materials):
    """Display the summary of selected materials and total cost."""
    stdscr.clear()
    stdscr.addstr(1, 0, "Summary of Selected Materials", curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.addstr(3, 0, "-" * curses.COLS)  # Line separating header from the content
    row = 5

    # Table columns
    stdscr.addstr(row, 0, f"{'Material':<30}{'Price per Unit (€)':<20}{'Unit Type':<15}{'Quantity':<20}{'Total Cost (€)':<20}", curses.A_BOLD)
    row += 1
    stdscr.addstr(row, 0, "-" * curses.COLS)  # Line separating header from content
    row += 1

    total_cost = 0.0
    for material, quantity in selected_materials.items():
        details = materials[material]
        price = details["price"]
        unit_type = details["unit_type"]
        total_material_cost = price * quantity
        total_cost += total_material_cost
        stdscr.addstr(row, 0, f"{material.capitalize():<30} €{price:<20.2f} {unit_type:<15} {quantity:<20} €{total_material_cost:<20.2f}")
        row += 1

    stdscr.addstr(row, 0, "-" * curses.COLS)  # Line separating materials and total cost
    row += 1
    stdscr.addstr(row, 0, f"Total Estimated Cost: €{total_cost:.2f}", curses.A_BOLD | curses.color_pair(4))

    stdscr.addstr(row + 2, 0, "Press any key to return to the main menu...", curses.A_BOLD)
    stdscr.refresh()

    stdscr.getch()  # Wait for a key press to return to main menu

def material_cost_estimator(stdscr):
    """Main function to handle the material cost estimator TUI logic."""
    stdscr.clear()
    setup_colors()

    # Load materials from the most recent CSV file or use defaults
    materials = load_materials_from_csv()

    selected_materials = {}  # Dictionary to store selected materials with quantities
    total_cost = 0.0  # Total cost of selected materials
    selected_button = 0  # Initially, "Select" is selected
    selected_material = 0  # Initially, first material is selected
    search_query = ""  # Search query input

    while True:
        # Draw the TUI table with available materials
        draw_table(stdscr, materials, selected_materials, selected_material, selected_button, total_cost, search_query)

        key = stdscr.getch()

        # Navigation: Up/Down arrows to move through materials
        if key == curses.KEY_UP:
            selected_material = (selected_material - 1) % len(materials)
        elif key == curses.KEY_DOWN:
            selected_material = (selected_material + 1) % len(materials)
        elif key == curses.KEY_RIGHT:
            selected_button = (selected_button + 1) % 7  # 7 buttons now (including exit)
        elif key == curses.KEY_LEFT:
            selected_button = (selected_button - 1) % 7
        elif key == ord('\n'):  # Enter key: Perform action based on selected button
            if selected_button == 0:  # Select material
                material_name = list(materials.keys())[selected_material]
                if material_name in selected_materials:
                    selected_materials.pop(material_name)
                else:
                    material_price = materials[material_name]["price"]
                    quantity = get_quantity_input(stdscr, material_name, material_price)
                    if quantity:
                        selected_materials[material_name] = selected_materials.get(material_name, 0) + quantity
                        total_cost += material_price * quantity
            elif selected_button == 1:  # Remove material
                material_name = list(materials.keys())[selected_material]
                if material_name in selected_materials:
                    selected_materials.pop(material_name)
                    total_cost = sum(material["price"] * quantity for material, quantity in selected_materials.items())
            elif selected_button == 2:  # Finish and view summary
                display_summary(stdscr, selected_materials, materials)
            elif selected_button == 3:  # Export to CSV
                save_materials_to_csv(materials)
            elif selected_button == 4:  # Add new material
                result = get_quantity_input(stdscr, "New Material", 0.0)  # You can further enhance this for material name and details
                if result:
                    material_name = f"NewMaterial{len(materials) + 1}"
                    price = 1.0
                    unit_type = "unit"
                    materials[material_name] = {"price": price, "unit_type": unit_type, "available_quantity": result}
            elif selected_button == 5:  # Exit
                save_materials_to_csv(materials)  # Save materials state before exiting
                break  # Exit the program

# Start the curses application
curses.wrapper(material_cost_estimator)

