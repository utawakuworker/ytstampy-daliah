import flet as ft
import os # Needed for path manipulation if refining file picker logic
from viewmodel.analysis_viewmodel import AnalysisViewModel
from typing import Dict, Any

class AppView(ft.UserControl):
    """
    The main View class for the Flet GUI.
    Displays the UI controls and interacts with the AnalysisViewModel.
    """
    def __init__(self, view_model: AnalysisViewModel):
        super().__init__()
        self.view_model = view_model
        # Assign the view's update method to the view model's callback
        # This is crucial for the ViewModel to trigger UI refreshes
        self.view_model.view_update_callback = self.update_view_state

        # --- Refs for UI elements that need updating or access ---
        # Status/Progress
        self.status_text_ref = ft.Ref[ft.Text]()
        self.progress_bar_ref = ft.Ref[ft.ProgressBar]()
        self.start_button_ref = ft.Ref[ft.ElevatedButton]()
        self.save_button_ref = ft.Ref[ft.ElevatedButton]() # Ref for save button
        self.visualize_button_ref = ft.Ref[ft.ElevatedButton]() # Ref for visualize button

        # Log/Results
        self.log_view_ref = ft.Ref[ft.ListView]()
        self.segments_table_ref = ft.Ref[ft.DataTable]()
        self.songs_table_ref = ft.Ref[ft.DataTable]()
        self.summary_text_ref = ft.Ref[ft.Text]()

        # Config Inputs (need refs to read values and potentially disable)
        self.source_type_ref = ft.Ref[ft.RadioGroup]()
        self.file_path_ref = ft.Ref[ft.TextField]()
        self.url_ref = ft.Ref[ft.TextField]()
        self.output_dir_ref = ft.Ref[ft.TextField]()
        self.sing_ref_time_ref = ft.Ref[ft.TextField]()
        self.non_sing_ref_time_ref = ft.Ref[ft.TextField]()
        self.ref_duration_ref = ft.Ref[ft.TextField]()
        self.hmm_threshold_ref = ft.Ref[ft.TextField]()
        self.min_segment_duration_ref = ft.Ref[ft.TextField]()
        self.min_segment_gap_ref = ft.Ref[ft.TextField]()
        self.n_components_ref = ft.Ref[ft.TextField]()
        self.min_duration_for_id_ref = ft.Ref[ft.TextField]()
        self.whisper_model_ref = ft.Ref[ft.Dropdown]()
        self.api_key_ref = ft.Ref[ft.TextField]()
        self.visualize_ref = ft.Ref[ft.Checkbox]()
        self.save_json_ref = ft.Ref[ft.Checkbox]()
        self.save_csv_ref = ft.Ref[ft.Checkbox]()
        # Refs for config section controls to disable them
        self.config_panel_ref = ft.Ref[ft.ExpansionPanel]()
        self.browse_file_button_ref = ft.Ref[ft.ElevatedButton]()
        self.select_dir_button_ref = ft.Ref[ft.ElevatedButton]()


        # File/Directory Pickers
        self.file_picker = ft.FilePicker(on_result=self._on_pick_file_result)
        self.directory_picker = ft.FilePicker(on_result=self._on_pick_directory_result)


    def build(self):
        """Builds the Flet UI control tree."""

        # --- Input Source Section ---
        input_source_section = ft.Card(
            content=ft.Container(
                padding=10,
                content=ft.Column([
                    ft.Text("Input Source", style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.RadioGroup(
                        ref=self.source_type_ref,
                        # Determine default based on initial ViewModel config
                        value="file" if self.view_model.get_config_value('file') else "url",
                        content=ft.Row([
                            ft.Radio(value="file", label="File"),
                            ft.Radio(value="url", label="URL"),
                        ])
                        # Add on_change later if needed to auto-clear other field
                    ),
                    ft.Row([
                        ft.TextField(
                            ref=self.file_path_ref,
                            label="File Path",
                            value=self.view_model.get_config_value('file', ''),
                            expand=True,
                            hint_text="Select audio file...",
                            # on_change=self._config_changed # Example if needed
                        ),
                        ft.ElevatedButton(
                            "Browse...",
                            ref=self.browse_file_button_ref,
                            on_click=lambda _: self.file_picker.pick_files(
                                allow_multiple=False,
                                allowed_extensions=["mp3", "wav", "m4a", "ogg", "flac"]
                            )
                        )
                    ]),
                    ft.TextField(
                        ref=self.url_ref,
                        label="YouTube URL",
                        value=self.view_model.get_config_value('url', ''),
                        expand=True,
                        hint_text="Enter YouTube video URL...",
                        # on_change=self._config_changed # Example if needed
                    ),
                ])
            )
        )

        # --- Configuration Section ---
        config_section = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.AMBER,
            elevation=4,
            divider_color=ft.colors.BLUE_GREY_100,
            controls=[
                ft.ExpansionPanel(
                    ref=self.config_panel_ref, # Ref to potentially disable header interaction
                    header=ft.ListTile(title=ft.Text("Configuration Parameters")),
                    content=ft.Container(
                        padding=ft.padding.only(left=15, right=15, bottom=10),
                        content=ft.Column([
                             # Row 1: Reference Times
                             ft.Row([
                                ft.TextField(ref=self.sing_ref_time_ref, label="Sing Ref Time (s)", value=str(self.view_model.get_config_value('singing_ref_time')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.non_sing_ref_time_ref, label="Non-Sing Ref (s)", value=str(self.view_model.get_config_value('non_singing_ref_time')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.ref_duration_ref, label="Ref Duration (s)", value=str(self.view_model.get_config_value('ref_duration')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                             # Row 2: Detection Params
                             ft.Row([
                                ft.TextField(ref=self.hmm_threshold_ref, label="HMM Thresh", value=str(self.view_model.get_config_value('hmm_threshold')), width=100, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.min_segment_duration_ref, label="Min Seg Dur (s)", value=str(self.view_model.get_config_value('min_segment_duration')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.min_segment_gap_ref, label="Min Seg Gap (s)", value=str(self.view_model.get_config_value('min_segment_gap')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.n_components_ref, label="PCA Comp", value=str(self.view_model.get_config_value('n_components')), width=100, keyboard_type=ft.KeyboardType.NUMBER),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            # Row 3: Identification Params
                            ft.Row([
                                ft.TextField(ref=self.min_duration_for_id_ref, label="Min ID Dur (s)", value=str(self.view_model.get_config_value('min_duration_for_id')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.Dropdown(ref=self.whisper_model_ref, label="Whisper Model", value=self.view_model.get_config_value('whisper_model'), width=150, options=[
                                    ft.dropdown.Option("tiny"), ft.dropdown.Option("base"), ft.dropdown.Option("small"),
                                    ft.dropdown.Option("medium"), ft.dropdown.Option("large")
                                ]),
                                ft.TextField(ref=self.api_key_ref, label="Gemini API Key", value=self.view_model.get_config_value('gemini_api_key'), password=True, can_reveal_password=True, expand=True),
                            ]),
                             # Row 4: Output Directory
                             ft.Row([
                                ft.TextField(ref=self.output_dir_ref, label="Output Directory", value=self.view_model.get_config_value('output_dir'), expand=True),
                                ft.ElevatedButton(
                                    "Select...",
                                    ref=self.select_dir_button_ref,
                                    on_click=lambda _: self.directory_picker.get_directory_path()
                                )
                            ]),
                            # Row 5: Output Options
                            ft.Row([
                                ft.Checkbox(ref=self.visualize_ref, label="Visualize", value=self.view_model.get_config_value('visualize')),
                                ft.Checkbox(ref=self.save_json_ref, label="Save JSON", value=self.view_model.get_config_value('save_results_json')),
                                ft.Checkbox(ref=self.save_csv_ref, label="Save CSV", value=self.view_model.get_config_value('save_results_dataframe')),
                            ])
                        ])
                    )
                )
            ]
        )

        # --- Controls Section ---
        controls_section = ft.Card(
            content=ft.Container(
                padding=10,
                content=ft.Column([
                     ft.Row([
                        ft.ElevatedButton(
                            "Start Analysis",
                            ref=self.start_button_ref,
                            icon=ft.icons.PLAY_ARROW,
                            on_click=self._start_analysis_clicked, # Connect action
                            width=200,
                            tooltip="Gather settings and start the analysis process"
                        ),
                        # Use indeterminate progress bar initially
                        ft.ProgressBar(ref=self.progress_bar_ref, value=None, width=300, bar_height=10, visible=False), # Start invisible
                     ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                     ft.Text("Idle", ref=self.status_text_ref, style=ft.TextThemeStyle.BODY_SMALL)
                ])
            )
        )

        # --- Results Section ---
        results_section = ft.Tabs(
            selected_index=0,
            animation_duration=300,
            tabs=[
                ft.Tab(
                    text="Detected Segments",
                    icon=ft.icons.MUSIC_NOTE,
                    content=ft.Column([
                        ft.DataTable(
                            ref=self.segments_table_ref,
                            columns=[
                                ft.DataColumn(ft.Text("#")), ft.DataColumn(ft.Text("Start (s)")),
                                ft.DataColumn(ft.Text("End (s)")), ft.DataColumn(ft.Text("Duration")),
                            ], rows=[], column_spacing=20, heading_row_height=35, data_row_max_height=40,
                        ),
                        ft.Text("Summary:", weight=ft.FontWeight.BOLD),
                        ft.Text("No analysis yet.", ref=self.summary_text_ref)
                    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
                ),
                ft.Tab(
                    text="Identified Songs", icon=ft.icons.QUEUE_MUSIC,
                     content=ft.DataTable(
                        ref=self.songs_table_ref,
                        columns=[
                            ft.DataColumn(ft.Text("Seg #")), ft.DataColumn(ft.Text("Title")),
                            ft.DataColumn(ft.Text("Artist")), ft.DataColumn(ft.Text("Confidence")),
                        ], rows=[], column_spacing=20, heading_row_height=35, data_row_max_height=40,
                    ),
                ),
                 ft.Tab( # Actions Tab
                    text="Actions", icon=ft.icons.SETTINGS_APPLICATIONS, # Changed icon
                    content=ft.Row([
                        ft.ElevatedButton(
                            "Save Results",
                            ref=self.save_button_ref,
                            icon=ft.icons.SAVE_ALT,
                            on_click=self._save_results_clicked, # Connect action
                            tooltip="Save detected segments and identified songs"
                        ),
                        ft.ElevatedButton(
                            "Visualize",
                            ref=self.visualize_button_ref,
                            icon=ft.icons.IMAGE,
                            on_click=self._visualize_clicked, # Connect action
                            tooltip="Show Matplotlib visualizations (if enabled)"
                        ),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=20) # Added spacing
                )
            ],
            expand=True,
        )

        # --- Log Section ---
        log_section = ft.Column([
                ft.Text("Log / Messages", style=ft.TextThemeStyle.TITLE_SMALL),
                ft.Container( # Wrap ListView in Container for border/padding
                    content=ft.ListView(ref=self.log_view_ref, height=150, spacing=1, auto_scroll=True),
                    border=ft.border.all(1, ft.colors.OUTLINE), # Add border
                    border_radius=ft.border_radius.all(5),
                    padding=5
                )
            ],
            expand=False # Keep fixed height
        )

        # --- Assemble Main Layout ---
        # This is the root control returned by build()
        return ft.Column(
            [
                input_source_section,
                config_section,
                controls_section,
                ft.Divider(height=10, color=ft.colors.TRANSPARENT),
                ft.Text("Results", style=ft.TextThemeStyle.TITLE_MEDIUM),
                ft.Container(results_section, expand=True, padding=ft.padding.only(top=5)),
                ft.Divider(height=5),
                log_section,
            ],
            expand=True,
            # scroll=ft.ScrollMode.ADAPTIVE # Let Flet handle scrolling if needed
        )

    def did_mount(self):
        """Called after the control is added to the page."""
        if self.page:
            self.page.overlay.extend([self.file_picker, self.directory_picker])
            self.page.update()
        print("AppView: did_mount")
        # Initial UI update based on ViewModel state
        self.update_view_state() # Ensure initial state is reflected


    def update_view_state(self):
        """
        Updates the UI elements based on the current state of the ViewModel.
        This method is called by the ViewModel's callback. Uses page.run_thread_safe
        if called from a background thread (which our ViewModel callback might be).
        """
        if not self.page or not self.controls:
             print("AppView: Skipping update, UI not ready.")
             return

        print("AppView: Updating view state...") # Debug

        # --- Get state from ViewModel ---
        is_running = self.view_model.is_analysis_running()
        status_msg = self.view_model.get_status_message()
        progress_val = self.view_model.get_progress_value()
        logs = self.view_model.get_log_messages()
        segments = self.view_model.get_detected_segments()
        id_results = self.view_model.get_identification_results()
        summary_info = self.view_model.get_summary_info()
        can_visualize = self.view_model.get_config_value('visualize', False)

        # --- Define UI update actions ---
        def _update_ui():
            # Status & Progress
            if self.status_text_ref.current:
                self.status_text_ref.current.value = status_msg
            if self.progress_bar_ref.current:
                self.progress_bar_ref.current.value = progress_val if is_running else None # Show indeterminate if not running but value is None? Or hide?
                self.progress_bar_ref.current.visible = is_running # Show only when running

            # Controls Enabled/Disabled State
            # Disable config inputs and start button while running
            # Enable save/visualize buttons only when *not* running and results exist
            enable_actions = not is_running and self.view_model.analysis_results is not None

            if self.start_button_ref.current: self.start_button_ref.current.disabled = is_running
            if self.save_button_ref.current: self.save_button_ref.current.disabled = not enable_actions
            if self.visualize_button_ref.current: self.visualize_button_ref.current.disabled = not (enable_actions and can_visualize)

            # Disable config section while running (more robustly)
            config_controls_to_disable = [
                self.source_type_ref, self.file_path_ref, self.url_ref, self.output_dir_ref,
                self.sing_ref_time_ref, self.non_sing_ref_time_ref, self.ref_duration_ref,
                self.hmm_threshold_ref, self.min_segment_duration_ref, self.min_segment_gap_ref,
                self.n_components_ref, self.min_duration_for_id_ref, self.whisper_model_ref,
                self.api_key_ref, self.visualize_ref, self.save_json_ref, self.save_csv_ref,
                self.browse_file_button_ref, self.select_dir_button_ref
            ]
            for control_ref in config_controls_to_disable:
                if control_ref.current:
                    control_ref.current.disabled = is_running
            # Also consider disabling the expansion panel header interaction if possible/needed

            # Log View
            if self.log_view_ref.current:
                # Optimize: Check if logs actually changed? For now, replace all.
                current_log_count = len(self.log_view_ref.current.controls)
                if len(logs) != current_log_count or (logs and self.log_view_ref.current.controls and logs[-1] != self.log_view_ref.current.controls[-1].value):
                     self.log_view_ref.current.controls = [ft.Text(msg, size=11, selectable=True) for msg in logs] # Make logs selectable
                     # Auto-scroll might need adjustment or explicit call after update
                     # self.log_view_ref.current.scroll_to(offset=-1, duration=100)

            # Segments Table
            if self.segments_table_ref.current:
                seg_rows = []
                if segments:
                    for i, (start, end) in enumerate(segments):
                        duration = end - start
                        seg_rows.append(ft.DataRow(cells=[
                            ft.DataCell(ft.Text(str(i + 1))), ft.DataCell(ft.Text(f"{start:.1f}")),
                            ft.DataCell(ft.Text(f"{end:.1f}")), ft.DataCell(ft.Text(f"{duration:.1f}s")),
                        ]))
                self.segments_table_ref.current.rows = seg_rows

            # Songs Table
            if self.songs_table_ref.current:
                song_rows = []
                if id_results:
                     # Simple sequential numbering for now
                     for i, item in enumerate(id_results):
                         title = item.get('identification', {}).get('title', 'N/A')
                         artist = item.get('identification', {}).get('artist', 'N/A')
                         conf = item.get('identification', {}).get('confidence', 'N/A')
                         seg_num_str = str(i + 1) # Placeholder
                         song_rows.append(ft.DataRow(cells=[
                             ft.DataCell(ft.Text(seg_num_str)), ft.DataCell(ft.Text(title or "N/A")),
                             ft.DataCell(ft.Text(artist or "N/A")), ft.DataCell(ft.Text(conf or "N/A")),
                         ]))
                self.songs_table_ref.current.rows = song_rows

            # Summary Text
            if self.summary_text_ref.current:
                self.summary_text_ref.current.value = summary_info["text"]

            # Trigger page update to redraw everything
            self.page.update()
            print("AppView: View state update complete.")

        # --- Execute the UI update ---
        # Check if we are on the main thread or a background thread
        # Flet doesn't provide a direct way to check, but updates from ViewModel
        # callbacks *should* be run safely using page.run_thread_safe if needed.
        # However, calling page.update() directly within the callback passed to
        # the ViewModel seems to work in many Flet scenarios. If issues arise,
        # wrap the _update_ui call: self.page.run_thread_safe(_update_ui)
        _update_ui()


    # --- UI Event Handlers ---

    def _start_analysis_clicked(self, e):
        """Handles the 'Start Analysis' button click."""
        print("AppView: Start Analysis clicked")
        # 1. Gather current config from UI fields
        current_ui_config = self._get_config_from_ui()
        if not current_ui_config.get('file') and not current_ui_config.get('url'):
             # Basic validation: Ensure input source is provided
             self.view_model._pipeline_status_update("Error: Please select a file or enter a URL.") # Use VM to update status
             return
        # 2. Update the ViewModel's config
        self.view_model.update_full_config(current_ui_config)
        # 3. Tell the ViewModel to start the analysis
        self.view_model.start_analysis()

    def _save_results_clicked(self, e):
        """Handles the 'Save Results' button click."""
        print("AppView: Save Results clicked")
        self.view_model.save_results()

    def _visualize_clicked(self, e):
        """Handles the 'Visualize' button click."""
        print("AppView: Visualize clicked")
        self.view_model.visualize_results()

    def _on_pick_file_result(self, e: ft.FilePickerResultEvent):
        """Handles the result from the file picker."""
        if e.files and len(e.files) > 0 and self.file_path_ref.current:
            selected_path = e.files[0].path
            self.file_path_ref.current.value = selected_path
            # Update ViewModel's config directly
            self.view_model.update_config_value('file', selected_path)
            self.view_model.update_config_value('url', None) # Clear URL
            # Update UI elements (radio button, URL field)
            if self.source_type_ref.current: self.source_type_ref.current.value = "file"
            if self.url_ref.current: self.url_ref.current.value = ""
            self.page.update() # Update relevant controls

    def _on_pick_directory_result(self, e: ft.FilePickerResultEvent):
        """Handles the result from the directory picker."""
        if e.path and self.output_dir_ref.current:
            selected_path = e.path
            self.output_dir_ref.current.value = selected_path
            self.view_model.update_config_value('output_dir', selected_path)
            self.page.update()

    # --- Helper to get config from UI ---

    def _get_config_from_ui(self) -> Dict[str, Any]:
        """Reads current values from UI controls and returns a config dict."""
        config = {}

        # Input Source
        source_type = self.source_type_ref.current.value if self.source_type_ref.current else "file"
        if source_type == "file":
            config['file'] = self.file_path_ref.current.value if self.file_path_ref.current else None
            config['url'] = None
        else:
            config['file'] = None
            config['url'] = self.url_ref.current.value if self.url_ref.current else None

        # Helper to safely get numeric values
        def get_numeric(ref, default, is_float=True):
            try:
                val_str = ref.current.value if ref.current else str(default)
                return float(val_str) if is_float else int(val_str)
            except (ValueError, TypeError):
                return default

        # Reference Segments
        config['singing_ref_time'] = get_numeric(self.sing_ref_time_ref, 1250)
        config['non_singing_ref_time'] = get_numeric(self.non_sing_ref_time_ref, 230)
        config['ref_duration'] = get_numeric(self.ref_duration_ref, 2.0)

        # Detection Parameters
        config['hmm_threshold'] = get_numeric(self.hmm_threshold_ref, 0.55)
        config['min_segment_duration'] = get_numeric(self.min_segment_duration_ref, 10.0)
        config['min_segment_gap'] = get_numeric(self.min_segment_gap_ref, 1.5)
        config['n_components'] = get_numeric(self.n_components_ref, 4, is_float=False)
        # config['dim_reduction'] = 'pca' # Assuming fixed for now

        # Identification Parameters
        config['min_duration_for_id'] = get_numeric(self.min_duration_for_id_ref, 30.0)
        config['whisper_model'] = self.whisper_model_ref.current.value if self.whisper_model_ref.current else 'base'
        config['gemini_api_key'] = self.api_key_ref.current.value if self.api_key_ref.current else ''

        # Output Settings
        config['output_dir'] = self.output_dir_ref.current.value if self.output_dir_ref.current else './output_analysis_gui'
        config['visualize'] = self.visualize_ref.current.value if self.visualize_ref.current else False
        config['save_results_json'] = self.save_json_ref.current.value if self.save_json_ref.current else False
        config['save_results_dataframe'] = self.save_csv_ref.current.value if self.save_csv_ref.current else False
        # config['results_json_file'] = 'song_identification_results.json' # Keep defaults from config.py
        # config['results_dataframe_file'] = 'song_identification_results.csv'

        return config 