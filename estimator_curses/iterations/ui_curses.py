import curses

# Sample materials data
materials = {
    "Cement": {"price": 5.0, "unit_type": "bags", "available_quantity": 100},
    "Sand": {"price": 3.0, "unit_type": "kgs", "available_quantity": 200},
    "Gravel": {"price": 8.0, "unit_type": "kgs", "available_quantity": 150},
    "Bricks": {"price": 0.25, "unit_type": "pieces", "available_quantity": 500},
    "Steel": {"price": 12.0, "unit_type": "kg", "available_quantity": 300},
}

# Function to draw the table
def draw_table(stdscr, selected_material_idx, selected_button_idx, search_query):
    height, width = stdscr.getmaxyx()

    # Header row
    stdscr.addstr(1, 0, "Material Cost Estimator", curses.A_BOLD | curses.A_UNDERLINE)

    # Search bar
    stdscr.addstr(3, 0, f"Search: {search_query}", curses.A_BOLD)

    # Table headers
    stdscr.addstr(5, 0, f"{'Material':<20}{'Price/Unit':<15}{'Unit Type':<15}{'Available Qty':<15}", curses.A_BOLD)
    stdscr.addstr(6, 0, "-" * width)  # Line separator

    # Filtered materials based on the search query
    row = 7
    for idx, (material, details) in enumerate(materials.items()):
        if search_query.lower() in material.lower():  # Filter based on search query
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
    buttons = ["Select", "Remove", "Finish"]
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

    selected_material_idx = 0  # Initially select the first material
    selected_button_idx = 0    # Initially select the first button
    search_query = ""          # No search query initially
    in_search_mode = True      # Flag to track if we're in search mode

    while True:
        # Draw the UI (only needs to be drawn once every key press)
        draw_table(stdscr, selected_material_idx, selected_button_idx, search_query)

        # Get user input
        key = stdscr.getch()

        # If we're in search mode, treat key presses as search input
        if in_search_mode:
            if key == curses.KEY_UP:
                selected_material_idx = (selected_material_idx - 1) % len(materials)
            elif key == curses.KEY_DOWN:
                selected_material_idx = (selected_material_idx + 1) % len(materials)
            elif key == curses.KEY_LEFT:
                selected_button_idx = (selected_button_idx - 1) % 3  # 3 buttons
            elif key == curses.KEY_RIGHT:
                selected_button_idx = (selected_button_idx + 1) % 3  # 3 buttons
            elif key == ord('\n'):  # Enter key: Select the current button
                if selected_button_idx == 0:  # "Select" button
                    selected_material = list(materials.keys())[selected_material_idx]
                    stdscr.addstr(15, 0, f"Selected {selected_material}. Press any key to continue.", curses.A_BOLD)
                    stdscr.refresh()
                    stdscr.getch()  # Wait for a key to continue
                    in_search_mode = False  # Stop search mode temporarily
                elif selected_button_idx == 1:  # "Remove" button
                    selected_material = list(materials.keys())[selected_material_idx]
                    # Remove the material from the materials dictionary
                    if selected_material in materials:
                        del materials[selected_material]
                        stdscr.addstr(15, 0, f"Removed {selected_material}. Press any key to continue.", curses.A_BOLD)
                    else:
                        stdscr.addstr(15, 0, f"Material {selected_material} not found. Press any key to continue.", curses.A_BOLD)
                    stdscr.refresh()
                    stdscr.getch()  # Wait for a key to continue
                    in_search_mode = False  # Stop search mode temporarily
                elif selected_button_idx == 2:  # "Finish" button
                    stdscr.addstr(15, 0, "Finished. Exiting... Press any key to exit.", curses.A_BOLD)
                    stdscr.refresh()
                    stdscr.getch()  # Wait for a key to continue
                    break  # Exit the program
            elif key == 27:  # Escape key to quit
                break
            elif key == curses.KEY_BACKSPACE or key == 127:  # Backspace key
                search_query = search_query[:-1]  # Remove the last character
            elif key != -1:  # Any other key will be considered a search character
                search_query += chr(key)  # Append the character to the search query
                if len(search_query) > 20:  # Limiting the length of the search query
                    search_query = search_query[-20:]
        
        # If we're not in search mode (waiting for "Press any key to continue"), we don't take search input
        if not in_search_mode:
            key = stdscr.getch()  # Wait for the user to press any key to continue
            in_search_mode = True  # Go back to search mode after continuing

        # Refresh only the updated sections of the screen
        stdscr.refresh()

# Start the program using curses wrapper
curses.wrapper(material_cost_estimator)

