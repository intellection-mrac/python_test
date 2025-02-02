import curses
import csv
import os
from datetime import datetime

# Default materials data
default_materials = {
    "Cement": {"price": 5.0, "unit_type": "bags", "available_quantity": 100},
    "Sand": {"price": 3.0, "unit_type": "kgs", "available_quantity": 200},
    "Gravel": {"price": 8.0, "unit_type": "kgs", "available_quantity": 150},
    "Bricks": {"price": 0.25, "unit_type": "pieces", "available_quantity": 500},
    "Steel": {"price": 12.0, "unit_type": "kg", "available_quantity": 300},
}

# Load materials from CSV
def load_materials_from_csv():
    try:
        # Get the latest CSV file based on timestamp
        csv_files = [f for f in os.listdir() if f.endswith(".csv")]
        if not csv_files:
            return default_materials  # Return default if no CSV found
        latest_file = max(csv_files, key=lambda f: os.path.getmtime(f))
        
        materials = {}
        with open(latest_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                materials[row['name']] = {
                    'price': float(row['price']),
                    'unit_type': row['unit_type'],
                    'available_quantity': int(row['available_quantity'])
                }
        return materials
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return default_materials  # Return default if loading fails

# Save materials to CSV
def save_materials_to_csv(materials):
    try:
        # Create a timestamped file name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"materials_{timestamp}.csv"
        
        with open(file_name, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["name", "price", "unit_type", "available_quantity"])
            writer.writeheader()
            for name, details in materials.items():
                writer.writerow({
                    'name': name,
                    'price': details['price'],
                    'unit_type': details['unit_type'],
                    'available_quantity': details['available_quantity']
                })
    except Exception as e:
        print(f"Error saving CSV: {e}")

# Function to draw the table
def draw_table(stdscr, selected_material_idx, selected_button_idx, search_query, materials):
    height, width = stdscr.getmaxyx()

    # Search bar
    stdscr.addstr(3, 0, f"Search: {search_query}", curses.A_BOLD)

    # Table headers
    stdscr.addstr(5, 0, f"{'Material':<20}{'Price/Unit':<15}{'Unit Type':<15}{'Available Qty':<15}", curses.A_BOLD)
    stdscr.addstr(6, 0, "-" * width)  # Line separator

    # Filtered materials based on the search query
    row = 7
    filtered_materials = [(material, details) for material, details in materials.items() if search_query.lower() in material.lower()]
    for idx, (material, details) in enumerate(filtered_materials):
        price = details["price"]
        unit_type = details["unit_type"]
        available_quantity = details["available_quantity"]

        # Highlight the selected material
        if idx == selected_material_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(row, 0, f"{material:<20}{price:<15}{unit_type:<15}{available_quantity:<15}")
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(row, 0, f"{material:<20}{price:<15}{unit_type:<15}{available_quantity:<15}")
        row += 1

    # Buttons section
    buttons = ["Select", "Remove", "Add", "Finish"]
    button_row = height - 4
    for i, button in enumerate(buttons):
        if i == selected_button_idx:
            stdscr.attron(curses.A_REVERSE)
            stdscr.addstr(button_row, i * 20, f"[ {button} ]")
            stdscr.attroff(curses.A_REVERSE)
        else:
            stdscr.addstr(button_row, i * 20, f"[ {button} ]")
    
    # Refresh only the updated sections of the screen
    stdscr.refresh()

# Function to handle the main program logic
def material_cost_estimator(stdscr):
    curses.curs_set(0)  # Hide the cursor
    stdscr.nodelay(1)   # Don't block on getch()

    # Load materials, either from CSV or use the default
    materials = load_materials_from_csv()

    selected_material_idx = 0  # Initially select the first material
    selected_button_idx = 0    # Initially select the first button
    search_query = ""          # No search query initially

    while True:
        # Draw the UI (only needs to be drawn once every key press)
        draw_table(stdscr, selected_material_idx, selected_button_idx, search_query, materials)

        # Get user input
        key = stdscr.getch()

        if key == ord('\n'):  # Enter key: Select the current button
            filtered_materials = [(material, details) for material, details in materials.items() if search_query.lower() in material.lower()]
            selected_material = filtered_materials[selected_material_idx] if filtered_materials else None
            
            if selected_button_idx == 0:  # "Select" button
                if selected_material:
                    stdscr.addstr(15, 0, f"Enter quantity for {selected_material[0]}: ", curses.A_BOLD)
                    stdscr.refresh()
                    curses.echo()
                    quantity = int(stdscr.getstr().decode('utf-8'))
                    stdscr.addstr(16, 0, f"Selected {selected_material[0]} with quantity {quantity}. Press any key to continue.", curses.A_BOLD)
                    stdscr.refresh()
                    stdscr.getch()  # Wait for a key to continue
            elif selected_button_idx == 1:  # "Remove" button
                if selected_material:
                    material_to_remove = selected_material[0]
                    if material_to_remove in materials:
                        del materials[material_to_remove]
                        stdscr.addstr(15, 0, f"Removed {material_to_remove}. Press any key to continue.", curses.A_BOLD)
                        stdscr.refresh()
                        stdscr.getch()  # Wait for a key to continue
            elif selected_button_idx == 2:  # "Add" button
                # Add a new material
                stdscr.addstr(15, 0, "Enter new material name: ", curses.A_BOLD)
                stdscr.refresh()
                curses.echo()
                new_name = stdscr.getstr().decode('utf-8')

                stdscr.addstr(16, 0, "Enter price per unit: ", curses.A_BOLD)
                stdscr.refresh()
                new_price = float(stdscr.getstr().decode('utf-8'))

                stdscr.addstr(17, 0, "Enter unit type (e.g., bags, kg): ", curses.A_BOLD)
                stdscr.refresh()
                new_unit = stdscr.getstr().decode('utf-8')

                stdscr.addstr(18, 0, "Enter available quantity: ", curses.A_BOLD)
                stdscr.refresh()
                new_quantity = int(stdscr.getstr().decode('utf-8'))

                materials[new_name] = {
                    "price": new_price,
                    "unit_type": new_unit,
                    "available_quantity": new_quantity
                }
                stdscr.addstr(19, 0, f"Added {new_name}. Press any key to continue.", curses.A_BOLD)
                stdscr.refresh()
                stdscr.getch()  # Wait for a key to continue
            elif selected_button_idx == 3:  # "Finish" button
                # Save to CSV before exiting
                save_materials_to_csv(materials)
                stdscr.addstr(15, 0, "Finished. Exiting... Press any key to exit.", curses.A_BOLD)
                stdscr.refresh()
                stdscr.getch()  # Wait for a key to continue
                break  # Exit the program

        elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace key
            search_query = search_query[:-1]  # Remove the last character
        elif key == 27:  # Escape key to quit
            break
        elif key != -1:  # Any other key will be considered a search character
            search_query += chr(key)  # Append the character to the search query
            if len(search_query) > 20:  # Limiting the length of the search query
                search_query = search_query[-20:]

        # Refresh only the updated sections of the screen
        stdscr.refresh()

# Start the program using curses wrapper
curses.wrapper(material_cost_estimator)

