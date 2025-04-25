import flet as ft
import os
import re # Import re for status translation
from viewmodel.analysis_viewmodel import AnalysisViewModel
from typing import Dict, Any, Optional, Callable

class AppView(ft.View):
    """
    The main View class for the Flet GUI.
    Displays the UI controls and interacts with the AnalysisViewModel.
    """
    def __init__(self, view_model: AnalysisViewModel, tr: Callable[[str, ...], str]):
        super().__init__()
        self.view_model = view_model
        self.tr = tr
        self.view_model.view_update_callback = self.update_view_state

        # --- Refs ---
        self.status_text_ref = ft.Ref[ft.Text]()
        self.progress_bar_ref = ft.Ref[ft.ProgressBar]()
        self.start_button_ref = ft.Ref[ft.ElevatedButton]()
        self.save_button_ref = ft.Ref[ft.ElevatedButton]()
        self.log_view_ref = ft.Ref[ft.ListView]()
        self.segments_table_ref = ft.Ref[ft.DataTable]()
        self.songs_table_ref = ft.Ref[ft.DataTable]()
        self.summary_text_ref = ft.Ref[ft.Text]()
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
        self.enable_hpss_ref = ft.Ref[ft.Checkbox]()
        self.api_key_ref = ft.Ref[ft.TextField]()
        self.visualize_ref = ft.Ref[ft.Checkbox]()
        self.save_json_ref = ft.Ref[ft.Checkbox]()
        self.save_csv_ref = ft.Ref[ft.Checkbox]()
        self.config_panel_ref = ft.Ref[ft.ExpansionPanel]()
        self.browse_file_button_ref = ft.Ref[ft.ElevatedButton]()
        self.select_dir_button_ref = ft.Ref[ft.ElevatedButton]()
        self.copy_comment_button_ref = ft.Ref[ft.ElevatedButton]()

        # --- Pickers ---
        self.file_picker = ft.FilePicker(on_result=self._on_pick_file_result)
        self.directory_picker = ft.FilePicker(on_result=self._on_pick_directory_result)

        # --- UI Building ---
        # Input Source Section
        input_source_section = ft.Card(
            content=ft.Container(
                padding=10,
                content=ft.Column([
                    ft.Text(self.tr("input_source_title"), style=ft.TextThemeStyle.TITLE_MEDIUM),
                    ft.RadioGroup(
                        ref=self.source_type_ref,
                        value="file" if self.view_model.get_config_value('file') else "url",
                        content=ft.Row([
                            ft.Radio(value="file", label=self.tr("file_option")),
                            ft.Radio(value="url", label=self.tr("url_option")),
                        ])
                    ),
                    ft.Row([
                        ft.TextField(
                            ref=self.file_path_ref, label=self.tr("file_path_label"),
                            value=self.view_model.get_config_value('file', ''), expand=True,
                            hint_text=self.tr("file_path_hint")
                        ),
                        ft.ElevatedButton(
                            self.tr("browse_button"), ref=self.browse_file_button_ref,
                            on_click=lambda _: self.file_picker.pick_files(
                                allow_multiple=False, allowed_extensions=["mp3", "wav", "m4a", "ogg", "flac"]
                            )
                        )
                    ]),
                    ft.TextField(
                        ref=self.url_ref, label=self.tr("youtube_url_label"),
                        value=self.view_model.get_config_value('url', ''), expand=True,
                        hint_text=self.tr("youtube_url_hint")
                    ),
                ])
            )
        )
        # Configuration Section
        config_section = ft.ExpansionPanelList(
            expand_icon_color=ft.colors.AMBER, elevation=4, divider_color=ft.colors.BLUE_GREY_100,
            controls=[
                ft.ExpansionPanel(
                    ref=self.config_panel_ref,
                    header=ft.ListTile(title=ft.Text(self.tr("config_title"))),
                    content=ft.Container(
                        padding=ft.padding.only(left=15, right=15, bottom=10, top=10),
                        content=ft.Column([
                             ft.Row([
                                ft.TextField(ref=self.sing_ref_time_ref, label=self.tr("sing_ref_time_label"), value=str(self.view_model.get_config_value('singing_ref_time')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.non_sing_ref_time_ref, label=self.tr("non_sing_ref_time_label"), value=str(self.view_model.get_config_value('non_singing_ref_time')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.ref_duration_ref, label=self.tr("ref_duration_label"), value=str(self.view_model.get_config_value('ref_duration')), width=150, keyboard_type=ft.KeyboardType.NUMBER),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                             ft.Row([
                                ft.TextField(ref=self.hmm_threshold_ref, label=self.tr("hmm_thresh_label"), value=str(self.view_model.get_config_value('hmm_threshold')), width=100, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.min_segment_duration_ref, label=self.tr("min_seg_dur_label"), value=str(self.view_model.get_config_value('min_segment_duration')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.min_segment_gap_ref, label=self.tr("min_seg_gap_label"), value=str(self.view_model.get_config_value('min_segment_gap')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.TextField(ref=self.n_components_ref, label=self.tr("pca_comp_label"), value=str(self.view_model.get_config_value('n_components')), width=100, keyboard_type=ft.KeyboardType.NUMBER),
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Row([
                                ft.TextField(ref=self.min_duration_for_id_ref, label=self.tr("min_id_dur_label"), value=str(self.view_model.get_config_value('min_duration_for_id')), width=120, keyboard_type=ft.KeyboardType.NUMBER),
                                ft.Dropdown(ref=self.whisper_model_ref, label=self.tr("whisper_model_label"), value=self.view_model.get_config_value('whisper_model'), width=150, options=[
                                    ft.dropdown.Option("tiny"), ft.dropdown.Option("base"), ft.dropdown.Option("small"), ft.dropdown.Option("medium"), ft.dropdown.Option("large") ]),
                                ft.Checkbox(ref=self.enable_hpss_ref, label=self.tr("enable_hpss_checkbox"), value=self.view_model.get_config_value('enable_hpss')),
                                ft.TextField(ref=self.api_key_ref, label=self.tr("gemini_api_key_label"), value=self.view_model.get_config_value('gemini_api_key'), password=True, can_reveal_password=True, expand=True),
                            ]),
                             ft.Row([
                                ft.TextField(ref=self.output_dir_ref, label=self.tr("output_dir_label"), value=self.view_model.get_config_value('output_dir'), expand=True),
                                ft.ElevatedButton(
                                    self.tr("select_button"), ref=self.select_dir_button_ref,
                                    on_click=lambda _: self.directory_picker.get_directory_path()
                                )
                            ]),
                            ft.Row([
                                ft.Checkbox(ref=self.save_json_ref, label=self.tr("save_json_checkbox"), value=self.view_model.get_config_value('save_results_json')),
                                ft.Checkbox(ref=self.save_csv_ref, label=self.tr("save_csv_checkbox"), value=self.view_model.get_config_value('save_results_dataframe')),
                            ], alignment=ft.MainAxisAlignment.START)
                        ], spacing=10)
                    )
                )
            ]
        )
        # Controls Section
        controls_section = ft.Card(
            content=ft.Container(
                padding=10,
                content=ft.Column([
                     ft.Row([
                        ft.ElevatedButton(
                            ref=self.start_button_ref, icon=ft.icons.PLAY_ARROW,
                            on_click=self._start_analysis_clicked, width=200,
                            tooltip=self.tr("start_button_tooltip") # Text set in update_view_state
                        ),
                        ft.ProgressBar(ref=self.progress_bar_ref, value=None, width=300, bar_height=10, visible=False),
                     ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                     ft.Text(self.tr("status_idle"), ref=self.status_text_ref, style=ft.TextThemeStyle.BODY_SMALL)
                ])
            )
        )
        # Results Section
        results_section = ft.Tabs(
            selected_index=0, animation_duration=300, expand=True,
            tabs=[
                ft.Tab(
                    text=self.tr("detected_segments_tab"), icon=ft.icons.MUSIC_NOTE,
                    content=ft.Column([
                         ft.Container(
                              content=ft.DataTable(
                                  ref=self.segments_table_ref,
                                  columns=[
                                      ft.DataColumn(ft.Text(self.tr("segment_col_num"))),
                                      ft.DataColumn(ft.Text(self.tr("segment_col_start"))),
                                      ft.DataColumn(ft.Text(self.tr("segment_col_end"))),
                                      ft.DataColumn(ft.Text(self.tr("segment_col_duration"))),
                                  ], rows=[], column_spacing=20, heading_row_height=35, data_row_max_height=40,
                              ), expand=True
                         ),
                         ft.Text(self.tr("summary_label"), weight=ft.FontWeight.BOLD),
                         ft.Text(self.tr("summary_no_analysis"), ref=self.summary_text_ref)
                    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
                ),
                ft.Tab(
                    text=self.tr("identified_songs_tab"), icon=ft.icons.QUEUE_MUSIC,
                     content=ft.DataTable(
                        ref=self.songs_table_ref,
                        columns=[
                            ft.DataColumn(ft.Text(self.tr("song_col_seg_num"))),
                            ft.DataColumn(ft.Text(self.tr("song_col_title"))),
                            ft.DataColumn(ft.Text(self.tr("song_col_artist"))),
                            ft.DataColumn(ft.Text(self.tr("song_col_confidence"))),
                        ], rows=[], column_spacing=20, heading_row_height=35, data_row_max_height=40,
                    ),
                ),
                 ft.Tab(
                    text=self.tr("actions_tab"), icon=ft.icons.SETTINGS_APPLICATIONS,
                    content=ft.Row([
                        ft.ElevatedButton(
                            self.tr("save_results_button"), ref=self.save_button_ref,
                            icon=ft.icons.SAVE_ALT, on_click=self._save_results_clicked,
                            tooltip=self.tr("save_button_tooltip")
                        ),
                        ft.ElevatedButton(
                            ref=self.copy_comment_button_ref,
                            text=self.tr("copy_comment_button"),
                            icon=ft.icons.COPY_ALL_OUTLINED,
                            tooltip=self.tr("copy_comment_tooltip"),
                            on_click=self._copy_youtube_comment_clicked,
                            disabled=True
                        ),
                    ], alignment=ft.MainAxisAlignment.CENTER, spacing=20)
                )
            ]
        )
        # Log Section
        log_section = ft.Column([
                ft.Text(self.tr("log_title"), style=ft.TextThemeStyle.TITLE_SMALL),
                ft.Container(
                    content=ft.ListView(ref=self.log_view_ref, height=150, spacing=1, auto_scroll=True),
                    border=ft.border.all(1, ft.colors.OUTLINE), border_radius=ft.border_radius.all(5), padding=5
                )
            ], expand=False
        )
        # Assign controls
        self.controls = [
            self.file_picker, self.directory_picker,
            ft.Column(
                [ input_source_section, config_section, controls_section, results_section, log_section ],
                spacing=10, expand=True
            )
        ]

    # --- Methods ---
    def did_mount(self):
        if self.page:
             self.page.overlay.append(self.file_picker)
             self.page.overlay.append(self.directory_picker)
             self.page.update()
        self.update_view_state()

    def update_view_state(self):
        if not self.page: return

        def _update_ui():
            # --- Completion Notification ---
            analysis_just_completed = self.view_model.check_and_reset_completion_flag()
            if analysis_just_completed:
                completion_snackbar = ft.SnackBar(
                    content=ft.Text(self.tr("status_completed_msg")), bgcolor=ft.colors.GREEN_ACCENT_700,
                )
                self.page.snack_bar = completion_snackbar
                self.page.snack_bar.open = True

            # --- Status & Progress ---
            status = self.view_model.get_status_message()
            progress = self.view_model.get_progress_value()
            is_running = self.view_model.is_analysis_running()
            display_status = self._translate_dynamic_status(status) # Use helper

            if self.status_text_ref.current: self.status_text_ref.current.value = display_status
            if self.progress_bar_ref.current:
                 self.progress_bar_ref.current.visible = is_running
                 self.progress_bar_ref.current.value = progress if progress is not None else None

            # --- Enable/Disable Controls & Update Button Text ---
            enable_controls = not is_running
            if self.start_button_ref.current:
                self.start_button_ref.current.disabled = is_running
                self.start_button_ref.current.text = self.tr("stop_button") if is_running else self.tr("start_button")
                self.start_button_ref.current.icon = ft.icons.STOP if is_running else ft.icons.PLAY_ARROW
                self.start_button_ref.current.tooltip = self.tr("stop_button_tooltip") if is_running else self.tr("start_button_tooltip")
                self.start_button_ref.current.on_click = self._stop_analysis_clicked if is_running else self._start_analysis_clicked

            results_exist = bool(self.view_model.get_detected_segments())
            if self.save_button_ref.current: self.save_button_ref.current.disabled = not enable_controls or not results_exist

            # Enable copy button only if analysis is not running AND songs were identified
            if self.copy_comment_button_ref.current:
                 self.copy_comment_button_ref.current.disabled = is_running or not bool(self.view_model.get_identified_songs())

            # Disable config inputs while running
            is_config_disabled = is_running
            if self.config_panel_ref.current:
                 # Disable individual controls within the panel if needed,
                 # or potentially disable the whole panel (though ExpansionPanel doesn't have a direct 'disabled' property)
                 # Disabling expansion might be an option if supported, or disable contained controls.
                 # Example: self.file_path_ref.current.disabled = is_config_disabled
                 pass # Implement disabling config elements as needed

            # --- Log/Results ---
            if self.log_view_ref.current:
                log_messages = self.view_model.get_log_messages()
                self.log_view_ref.current.controls.clear()
                for msg in log_messages:
                    display_msg = self._translate_dynamic_log(msg) # Use helper
                    self.log_view_ref.current.controls.append(ft.Text(display_msg, size=12))

            # --- Update Tables ---
            if self.segments_table_ref.current:
                detected_segments = self.view_model.get_detected_segments()
                self.segments_table_ref.current.rows.clear()
                for i, seg in enumerate(detected_segments):
                    self.segments_table_ref.current.rows.append(
                        ft.DataRow(cells=[
                            ft.DataCell(ft.Text(str(i + 1))),
                            ft.DataCell(ft.Text(f"{seg.start:.2f}")),
                            ft.DataCell(ft.Text(f"{seg.end:.2f}")),
                            ft.DataCell(ft.Text(f"{seg.duration:.2f}")),
                        ])
                    )

            if self.songs_table_ref.current:
                 identified_songs = self.view_model.get_identified_songs()
                 self.songs_table_ref.current.rows.clear()
                 for i, song_data in enumerate(identified_songs):
                     display_seg_num = str(i + 1)
                     title = song_data.get("title") or self.tr("unknown_title_placeholder")
                     artist = song_data.get("artist") or self.tr("unknown_artist_placeholder")
                     confidence = song_data.get("confidence", "-")
                     self.songs_table_ref.current.rows.append(
                         ft.DataRow(cells=[
                             ft.DataCell(ft.Text(display_seg_num)),
                             ft.DataCell(ft.Text(title)),
                             ft.DataCell(ft.Text(artist)),
                             ft.DataCell(ft.Text(confidence.capitalize())),
                         ])
                     )

            # --- Update Summary Text ---
            if self.summary_text_ref.current:
                 summary_info = self.view_model.get_summary_info()
                 summary_key = summary_info.get("key", "summary_no_analysis")
                 summary_text = self.tr(summary_key, **summary_info.get("kwargs", {}))
                 self.summary_text_ref.current.value = summary_text

            # --- Final Page Update ---
            self.page.update()

        if self.page: _update_ui()

    # --- Translation Helpers for Dynamic Content ---
    def _translate_dynamic_status(self, status: str) -> str:
        """Translates status messages that might contain placeholders or numbers."""
        # Simple regex to find common patterns like (Elapsed: HH:MM:SS) or numbers
        # More complex patterns might need more robust parsing
        match_elapsed = re.search(r"\(Elapsed: (\d{2,}:\d{2}:\d{2}|\d{1,2}:\d{2})\)", status)
        if match_elapsed:
            status = status[:match_elapsed.start()].strip()

        # Translate the raw status part using keys
        key_map = {
            "Idle": "status_idle",
            "Initializing...": "status_initializing",
            "Extracting features...": "status_extracting_features",
            "Detecting singing segments (HMM)...": "status_detecting_segments",
            "Processing and refining segments...": "status_processing_segments",
            "Identifying songs in segments...": "status_identifying_songs",
            "Starting Analysis Pipeline...": "status_pipeline_start",
            "Analysis pipeline completed.": "status_pipeline_complete",
            "Saving results...": "status_saving_results",
            "Generating visualization files...": "status_visualizing",
            # Add more mappings as needed
        }
        # Handle "Loading audio source: ..." separately
        loading_prefix = "Loading audio source:"
        if status.startswith(loading_prefix):
            source = status[len(loading_prefix):].strip()
            display_status = f"{self.tr('status_loading_audio')} {source}"
        else:
            # Translate using map or keep raw if unknown
            translation_key = key_map.get(status)
            display_status = self.tr(translation_key) if translation_key else status

        return display_status

    def _translate_dynamic_log(self, msg: str) -> str:
        """Translates log messages, similar to status."""
        # Example: "[STATUS] Analysis completed successfully."
        # Extract the core message after the prefix like [STATUS], [INFO], [ERROR]
        prefix_match = re.match(r"^\[([A-Z]+)\]\s*(.*)", msg)
        if prefix_match:
            core_message = prefix_match.group(2)
            # Translate the core message using keys
            key_map = {
                "Application started.": "log_app_started",
                "Loaded configuration from": "log_config_loaded",
                "No saved configuration found, using defaults.": "log_config_default",
                "Stop requested by user.": "log_stop_requested",
                # Add more mappings as needed
            }
            translation_key = key_map.get(core_message)
            if translation_key:
                return self.tr(translation_key)
            else:
                return core_message # Return original if no key found
        else:
            return msg # Return original if no prefix match

    # --- Callbacks & Config ---
    def _start_analysis_clicked(self, e):
        if self.view_model.is_analysis_running():
             print(self.tr("log_stop_requested"))
             self.view_model.request_stop()
        else:
             config_data = self._get_config_from_ui()
             self.view_model.update_full_config(config_data)
             self.view_model.start_analysis()

    def _stop_analysis_clicked(self, e):
        """Handles the Stop Analysis button click."""
        # Optional: Add a print or log for confirmation
        print("View: Stop button clicked, requesting stop...")
        if self.view_model:
            self.view_model.request_stop()

    def _save_results_clicked(self, e): self.view_model.save_results()

    def _copy_youtube_comment_clicked(self, e):
        """Handles the Copy YouTube Comment button click."""
        if not self.page: return

        comment_string = self.view_model.get_youtube_comment_string()

        if comment_string:
            self.page.set_clipboard(comment_string)
            snack_bar_msg = self.tr("comment_copied_success")
            snack_bar_color = ft.colors.GREEN
        else:
            snack_bar_msg = self.tr("comment_copied_empty")
            snack_bar_color = ft.colors.AMBER # Or use an info color

        snack_bar = ft.SnackBar(
            content=ft.Text(snack_bar_msg),
            bgcolor=snack_bar_color
        )
        self.page.snack_bar = snack_bar
        self.page.snack_bar.open = True
        self.page.update()

    def _on_pick_file_result(self, e: ft.FilePickerResultEvent):
        if e.files:
            selected_file = e.files[0].path
            if self.file_path_ref.current:
                self.file_path_ref.current.value = selected_file
                if self.source_type_ref.current: self.source_type_ref.current.value = "file"
                self.page.update()

    def _on_pick_directory_result(self, e: ft.FilePickerResultEvent):
        if e.path:
            selected_dir = e.path
            if self.output_dir_ref.current:
                self.output_dir_ref.current.value = selected_dir
                self.page.update()

    def _get_config_from_ui(self) -> Dict[str, Any]:
        """Gathers current configuration values from the UI controls."""
        config = {}
        def get_numeric(ref, default, is_float=True):
             val_str = ref.current.value if ref.current else str(default)
             label_key = "" # Find the translation key for the label
             if ref == self.sing_ref_time_ref: label_key = "sing_ref_time_label"
             elif ref == self.non_sing_ref_time_ref: label_key = "non_sing_ref_time_label"
             elif ref == self.ref_duration_ref: label_key = "ref_duration_label"
             elif ref == self.hmm_threshold_ref: label_key = "hmm_thresh_label"
             elif ref == self.min_segment_duration_ref: label_key = "min_seg_dur_label"
             elif ref == self.min_segment_gap_ref: label_key = "min_seg_gap_label"
             elif ref == self.n_components_ref: label_key = "pca_comp_label"
             elif ref == self.min_duration_for_id_ref: label_key = "min_id_dur_label"
             else: label_key = ref.current.label if ref.current else "unknown_field"
             try: return float(val_str) if is_float else int(val_str)
             except (ValueError, TypeError):
                 print(self.tr("log_invalid_numeric", value=val_str, default=default, label=self.tr(label_key)))
                 if ref.current: ref.current.value = str(default)
                 if self.page: self.page.update()
                 return default

        source_type = self.source_type_ref.current.value if self.source_type_ref.current else "file"
        config['file'] = self.file_path_ref.current.value if source_type == "file" and self.file_path_ref.current else None
        config['url'] = self.url_ref.current.value if source_type == "url" and self.url_ref.current else None
        config['output_dir'] = self.output_dir_ref.current.value if self.output_dir_ref.current else "."
        config['singing_ref_time'] = get_numeric(self.sing_ref_time_ref, 10.0)
        config['non_singing_ref_time'] = get_numeric(self.non_sing_ref_time_ref, 10.0)
        config['ref_duration'] = get_numeric(self.ref_duration_ref, 2.0)
        config['hmm_threshold'] = get_numeric(self.hmm_threshold_ref, 0.5)
        config['min_segment_duration'] = get_numeric(self.min_segment_duration_ref, 5.0)
        config['min_segment_gap'] = get_numeric(self.min_segment_gap_ref, 1.0)
        config['n_components'] = get_numeric(self.n_components_ref, 4, is_float=False)
        config['min_duration_for_id'] = get_numeric(self.min_duration_for_id_ref, 10.0)
        config['whisper_model'] = self.whisper_model_ref.current.value if self.whisper_model_ref.current else "base"
        config['enable_hpss'] = self.enable_hpss_ref.current.value if self.enable_hpss_ref.current else True
        config['gemini_api_key'] = self.api_key_ref.current.value if self.api_key_ref.current else ""
        config['save_results_json'] = self.save_json_ref.current.value if self.save_json_ref.current else False
        config['save_results_dataframe'] = self.save_csv_ref.current.value if self.save_csv_ref.current else False
        return config 