import curses
import csv

def setup_colors():
    """Initialize color pairs for the TUI interface."""
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)  # Blue header
    curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_WHITE)  # White background
    curses.init_pair(3, curses.COLOR_BLACK, curses.COLOR_YELLOW)  # Highlighted row
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_GREEN)  # Success
    curses.init_pair(5, curses.COLOR_RED, curses.COLOR_WHITE)  # Error
    curses.init_pair(7, curses.COLOR_WHITE, curses.COLOR_CYAN)  # Button color

def draw_table(stdscr, materials, selected_materials, selected_material, selected_button, total_cost, search_query):
    """Display the list of available materials and real-time total cost."""
    stdscr.clear()

    # Header row
    stdscr.attron(curses.color_pair(1))  # Header color
    stdscr.addstr(1, 0, "Material Inventory", curses.A_BOLD | curses.A_UNDERLINE)
    stdscr.attroff(curses.color_pair(1))
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
    for idx, (material, (price, unit_type, available_quantity)) in enumerate(materials.items()):
        # Highlight the material name if it matches the search query
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
        ("Add Material", "Add new material to inventory"),
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

def get_quantity_input(stdscr, material_name, material_price):
    """Prompt the user to enter a quantity for the selected material."""
    stdscr.clear()
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, f"Selected material: {material_name.capitalize()} - €{material_price:.2f} per unit", curses.A_BOLD)
    stdscr.attroff(curses.color_pair(1))
    stdscr.addstr(2, 0, f"Enter quantity for {material_name.capitalize()}: ")
    stdscr.refresh()

    curses.echo()  # Enable echoing of input characters
    quantity_str = stdscr.getstr(3, 0, 10).decode('utf-8')  # Get user input for quantity
    curses.noecho()  # Disable echoing after input is received

    # Validate that the input is a positive float
    try:
        quantity = float(quantity_str)
        if quantity <= 0:
            raise ValueError("Quantity must be a positive number.")
        return quantity
    except ValueError:
        stdscr.clear()
        stdscr.attron(curses.color_pair(5))  # Error message color
        stdscr.addstr(0, 0, "Invalid quantity entered. Please enter a valid positive number.", curses.A_BOLD)
        stdscr.attroff(curses.color_pair(5))
        stdscr.refresh()
        stdscr.getch()  # Wait for user input before returning
        return None

def get_material_input(stdscr):
    """Prompt the user to enter a new material with price, unit, and available quantity."""
    stdscr.clear()
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, 0, "Add New Material", curses.A_BOLD)
    stdscr.attroff(curses.color_pair(1))

    stdscr.addstr(2, 0, "Enter material name: ")
    stdscr.refresh()
    curses.echo()
    material_name = stdscr.getstr(3, 0, 20).decode('utf-8')
    curses.noecho()

    stdscr.addstr(4, 0, "Enter price per unit (€): ")
    stdscr.refresh()
    curses.echo()
    price_str = stdscr.getstr(5, 0, 10).decode('utf-8')
    curses.noecho()

    stdscr.addstr(6, 0, "Enter unit type (e.g., bags, kgs, running meters): ")
    stdscr.refresh()
    curses.echo()
    unit_type = stdscr.getstr(7, 0, 20).decode('utf-8')
    curses.noecho()

    stdscr.addstr(8, 0, "Enter available quantity: ")
    stdscr.refresh()
    curses.echo()
    available_quantity_str = stdscr.getstr(9, 0, 10).decode('utf-8')
    curses.noecho()

    try:
        price = float(price_str)
        available_quantity = int(available_quantity_str)
        if price <= 0 or available_quantity < 0:
            raise ValueError("Price and available quantity must be positive numbers.")
        return material_name, price, unit_type, available_quantity
    except ValueError:
        stdscr.clear()
        stdscr.attron(curses.color_pair(5))  # Error message color
        stdscr.addstr(0, 0, "Invalid input. Please enter valid price and available quantity.", curses.A_BOLD)
        stdscr.attroff(curses.color_pair(5))
        stdscr.refresh()
        stdscr.getch()
        return None

def display_summary(stdscr, selected_materials, materials):
    """Display the summary of selected materials and the total cost."""
    stdscr.clear()
    stdscr.attron(curses.color_pair(1))  # Header color
    stdscr.addstr(0, 0, "Selected Materials and their Costs:", curses.A_BOLD)
    stdscr.attroff(curses.color_pair(1))

    total_cost = 0
    row = 2
    # Display selected materials with their quantities and calculated costs
    for material, quantity in selected_materials.items():
        price, unit_type, _ = materials[material]
        material_cost = price * quantity
        total_cost += material_cost
        stdscr.addstr(row, 0, f"{material.capitalize():<30} {quantity:<10} {unit_type:<10} - €{material_cost:.2f}")
        row += 1

    # Display total cost
    stdscr.attron(curses.color_pair(4))  # Total cost color
    stdscr.addstr(row, 0, f"\nTotal estimated cost: €{total_cost:.2f}", curses.A_BOLD)
    stdscr.attroff(curses.color_pair(4))
    stdscr.addstr(row + 2, 0, "Press any key to exit.")
    stdscr.refresh()
    stdscr.getch()  # Wait for user to press any key to exit

def export_inventory_to_csv(stdscr, materials, selected_materials):
    """Export the remaining inventory to a CSV file."""
    remaining_materials = {material: (price, unit_type) for material, (price, unit_type, _) in materials.items() if material not in selected_materials}

    try:
        # Write remaining materials to CSV
        with open("remaining_materials.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Material", "Price per Unit (€)", "Unit Type"])
            for material, (price, unit_type) in remaining_materials.items():
                writer.writerow([material, price, unit_type])

        # Inform the user that the file has been created successfully
        stdscr.clear()
        stdscr.attron(curses.color_pair(4))  # Success message color
        stdscr.addstr(0, 0, "Remaining material inventory exported to 'remaining_materials.csv'.", curses.A_BOLD)
        stdscr.attroff(curses.color_pair(4))
        stdscr.refresh()
        stdscr.getch()  # Wait for user to press a key before returning

    except Exception as e:
        # Handle any errors that occur while writing to CSV
        stdscr.clear()
        stdscr.attron(curses.color_pair(5))  # Error message color
        stdscr.addstr(0, 0, f"Error exporting to CSV: {e}", curses.A_BOLD)
        stdscr.attroff(curses.color_pair(5))
        stdscr.refresh()
        stdscr.getch()

def material_cost_estimator(stdscr):
    """Main function to handle the material cost estimator TUI logic."""
    stdscr.clear()
    setup_colors()

    # Initial materials database (hardcoded for simplicity)
    materials = {
        "cement": (12.50, "bags", 5000),
        "sand": (20.00, "kgs", 1000),
        "gravel": (15.00, "kgs", 3000)
    }

    selected_materials = {}  # Dictionary to store selected materials with quantities
    total_cost = 0.0  # Total cost of selected materials
    selected_button = 0  # Initially, "Select" is selected
    selected_material = 0  # Initially, first material is selected
    search_query = ""  # Search query input

    while True:
        # Draw the TUI table with available materials
        draw_table(stdscr, materials, selected_materials, selected_material, selected_button, total_cost, search_query)

        key = stdscr.getch()

        if key == curses.KEY_UP:
            selected_material = (selected_material - 1) % len(materials)
        elif key == curses.KEY_DOWN:
            selected_material = (selected_material + 1) % len(materials)
        elif key == curses.KEY_RIGHT:
            selected_button = (selected_button + 1) % 6  # 6 buttons now (including exit)
        elif key == curses.KEY_LEFT:
            selected_button = (selected_button - 1) % 6
        elif key == ord('\n'):  # Enter key: Perform action based on selected button
            if selected_button == 0:  # Select material
                material_name = list(materials.keys())[selected_material]
                if material_name in selected_materials:
                    selected_materials.pop(material_name)
                else:
                    material_price, unit_type, available_quantity = materials[material_name]
                    quantity = get_quantity_input(stdscr, material_name, material_price)
                    if quantity:
                        selected_materials[material_name] = selected_materials.get(material_name, 0) + quantity
                        total_cost += material_price * quantity
            elif selected_button == 1:  # Remove material
                material_name = list(materials.keys())[selected_material]
                if material_name in selected_materials:
                    selected_materials.pop(material_name)
                    total_cost = sum(material_price * quantity for material_name, quantity in selected_materials.items())
            elif selected_button == 2:  # Finish and view summary
                display_summary(stdscr, selected_materials, materials)
            elif selected_button == 3:  # Export to CSV
                export_inventory_to_csv(stdscr, materials, selected_materials)
            elif selected_button == 4:  # Add new material
                result = get_material_input(stdscr)
                if result:
                    material_name, price, unit_type, available_quantity = result
                    materials[material_name] = (price, unit_type, available_quantity)  # Add new material with available quantity
            elif selected_button == 5:  # Exit
                break

        elif key == ord('q'):  # Quick exit with 'q'
            break

        elif key == curses.KEY_BACKSPACE or key == 127:  # Handle backspace key
            search_query = search_query[:-1]
        elif key == curses.KEY_DC:  # Handle delete key
            search_query = ""
        elif 32 <= key <= 126:  # Handle normal character input for search
            search_query += chr(key)

if __name__ == "__main__":
    curses.wrapper(material_cost_estimator)

