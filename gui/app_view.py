"""
app_view.py
------------
Main Flet GUI view for the Singing Analysis application.
Displays UI controls, binds to the AnalysisViewModel, and handles user interaction.
All translation key maps are module-level constants for maintainability.
"""
import re
from typing import Any, Callable, Dict, List

import flet as ft

from viewmodel.analysis_viewmodel import AnalysisViewModel

# --- Translation Key Maps ---
STATUS_KEY_MAP = {
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
}
LOG_KEY_MAP = {
    "Application started.": "log_app_started",
    "Loaded configuration from": "log_config_loaded",
    "No saved configuration found, using defaults.": "log_config_default",
    "Stop requested by user.": "log_stop_requested",
}

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
        ref_names = [
            "status_text", "progress_bar", "start_button", "save_button", "log_view", "segments_table", "songs_table",
            "summary_text", "source_type", "file_path", "url", "output_dir", "sing_ref_time", "non_sing_ref_time",
            "ref_duration", "hmm_threshold", "min_segment_duration", "min_segment_gap", "n_components",
            "min_duration_for_id", "whisper_model", "enable_hpss", "api_key", "visualize", "save_json", "save_csv",
            "config_panel", "browse_file_button", "select_dir_button", "copy_comment_button"
        ]
        for name in ref_names:
            setattr(self, f"{name}_ref", ft.Ref())
        # --- Pickers ---
        self.file_picker = ft.FilePicker(on_result=self._on_pick_file_result)
        self.directory_picker = ft.FilePicker(on_result=self._on_pick_directory_result)

        # --- UI Building ---
        input_source_section = self._build_input_source_section()
        config_section = self._build_config_section()
        controls_section = self._build_controls_section()
        results_section = self._build_results_section()
        log_section = self._build_log_section()
        self.controls = [
            self.file_picker, self.directory_picker,
            ft.Column(
                [input_source_section, config_section, controls_section, results_section, log_section],
                spacing=10, expand=True
            )
        ]

        # Helper for enabling/disabling controls
        def set_disabled(refs: List[ft.Ref], disabled: bool = True) -> None:
            for ref in refs:
                if ref.current:
                    ref.current.disabled = disabled
        self.set_disabled = set_disabled

    # --- Properties for frequently accessed state ---
    @property
    def detected_segments(self) -> List[Any]:
        return self.view_model.get_detected_segments()

    @property
    def identified_songs(self) -> List[Dict[str, Any]]:
        return self.view_model.get_identified_songs()

    @property
    def config_refs(self) -> List[ft.Ref]:
        return [
            self.file_path_ref, self.url_ref, self.output_dir_ref, self.sing_ref_time_ref, self.non_sing_ref_time_ref,
            self.ref_duration_ref, self.hmm_threshold_ref, self.min_segment_duration_ref, self.min_segment_gap_ref,
            self.n_components_ref, self.min_duration_for_id_ref, self.whisper_model_ref, self.enable_hpss_ref,
            self.api_key_ref, self.visualize_ref, self.save_json_ref, self.save_csv_ref
        ]

    @property
    def enable_controls(self) -> bool:
        return not self.view_model.is_analysis_running()

    # --- UI Section Helpers ---
    def _build_input_source_section(self) -> ft.Card:
        return ft.Card(
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
                            hint_text=self.tr("file_path_hint"), tooltip=self.tr("file_path_tooltip")
                        ),
                        ft.ElevatedButton(
                            self.tr("browse_button"), ref=self.browse_file_button_ref,
                            on_click=lambda _: self.file_picker.pick_files(
                                allow_multiple=False, allowed_extensions=["mp3", "wav", "m4a", "ogg", "flac"]
                            ),
                            tooltip=self.tr("browse_button_tooltip")
                        )
                    ]),
                    ft.TextField(
                        ref=self.url_ref, label=self.tr("youtube_url_label"),
                        value=self.view_model.get_config_value('url', ''), expand=True,
                        hint_text=self.tr("youtube_url_hint"), tooltip=self.tr("youtube_url_tooltip")
                    ),
                ])
            )
        )

    def _build_config_section(self) -> ft.ExpansionPanelList:
        def make_config_field(ref, config_key, label_key, width=150, keyboard_type=ft.KeyboardType.NUMBER, is_dropdown=False, dropdown_options=None, is_checkbox=False, is_password=False, expand=False):
            if is_dropdown:
                return ft.Dropdown(
                    ref=ref,
                    label=self.tr(label_key),
                    value=self.view_model.get_config_value(config_key),
                    width=width,
                    options=[ft.dropdown.Option(opt) for opt in dropdown_options],
                    tooltip=self.tr(label_key + "_tooltip")
                )
            elif is_checkbox:
                return ft.Checkbox(
                    ref=ref,
                    label=self.tr(label_key),
                    value=self.view_model.get_config_value(config_key),
                    tooltip=self.tr(label_key + "_tooltip")
                )
            else:
                return ft.TextField(
                    ref=ref,
                    label=self.tr(label_key),
                    value=str(self.view_model.get_config_value(config_key)),
                    width=width,
                    keyboard_type=keyboard_type,
                    password=is_password,
                    can_reveal_password=is_password,
                    expand=expand,
                    tooltip=self.tr(label_key + "_tooltip")
                )
        config_fields_row1 = [
            (self.sing_ref_time_ref, 'singing_ref_time', 'sing_ref_time_label'),
            (self.non_sing_ref_time_ref, 'non_singing_ref_time', 'non_sing_ref_time_label'),
            (self.ref_duration_ref, 'ref_duration', 'ref_duration_label'),
        ]
        config_fields_row2 = [
            (self.hmm_threshold_ref, 'hmm_threshold', 'hmm_thresh_label', 100),
            (self.min_segment_duration_ref, 'min_segment_duration', 'min_seg_dur_label', 120),
            (self.min_segment_gap_ref, 'min_segment_gap', 'min_seg_gap_label', 120),
            (self.n_components_ref, 'n_components', 'pca_comp_label', 100),
        ]
        config_fields_row3 = [
            (self.min_duration_for_id_ref, 'min_duration_for_id', 'min_id_dur_label', 120),
        ]
        return ft.ExpansionPanelList(
            expand_icon_color=ft.colors.AMBER, elevation=4, divider_color=ft.colors.BLUE_GREY_100,
            controls=[
                ft.ExpansionPanel(
                    ref=self.config_panel_ref,
                    header=ft.ListTile(title=ft.Text(self.tr("config_title"))),
                    content=ft.Container(
                        padding=ft.padding.only(left=15, right=15, bottom=10, top=10),
                        content=ft.Column([
                            ft.Row([
                                *[make_config_field(ref, key, label) for ref, key, label in config_fields_row1]
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Row([
                                *[make_config_field(ref, key, label, width) for ref, key, label, width in config_fields_row2]
                            ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                            ft.Row([
                                make_config_field(*config_fields_row3[0]),
                                make_config_field(self.whisper_model_ref, 'whisper_model', 'whisper_model_label', 150, is_dropdown=True, dropdown_options=["tiny", "base", "small", "medium", "large"]),
                                make_config_field(self.enable_hpss_ref, 'enable_hpss', 'enable_hpss_checkbox', is_checkbox=True),
                                make_config_field(self.api_key_ref, 'gemini_api_key', 'gemini_api_key_label', expand=True, is_password=True),
                            ]),
                            ft.Row([
                                make_config_field(self.output_dir_ref, 'output_dir', 'output_dir_label', expand=True),
                                ft.ElevatedButton(
                                    self.tr("select_button"), ref=self.select_dir_button_ref,
                                    on_click=lambda _: self.directory_picker.get_directory_path(),
                                    tooltip=self.tr("select_button_tooltip")
                                )
                            ]),
                            ft.Row([
                                make_config_field(self.save_json_ref, 'save_results_json', 'save_json_checkbox', is_checkbox=True),
                                make_config_field(self.save_csv_ref, 'save_results_dataframe', 'save_csv_checkbox', is_checkbox=True),
                            ], alignment=ft.MainAxisAlignment.START)
                        ], spacing=10)
                    )
                )
            ]
        )

    def _build_controls_section(self) -> ft.Card:
        return ft.Card(
            content=ft.Container(
                padding=10,
                content=ft.Column([
                     ft.Row([
                        ft.ElevatedButton(
                            ref=self.start_button_ref, icon=ft.icons.PLAY_ARROW,
                            on_click=self._start_analysis_clicked, width=200,
                            tooltip=self.tr("start_button_tooltip")
                        ),
                        ft.ProgressBar(ref=self.progress_bar_ref, value=None, width=300, bar_height=10, visible=False),
                     ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
                     ft.Text(self.tr("status_idle"), ref=self.status_text_ref, style=ft.TextThemeStyle.BODY_SMALL)
                ])
            )
        )

    def _build_results_section(self) -> ft.Tabs:
        return ft.Tabs(
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

    def _build_log_section(self) -> ft.Column:
        return ft.Column([
                ft.Text(self.tr("log_title"), style=ft.TextThemeStyle.TITLE_SMALL),
                ft.Container(
                    content=ft.ListView(ref=self.log_view_ref, height=150, spacing=1, auto_scroll=True),
                    border=ft.border.all(1, ft.colors.OUTLINE), border_radius=ft.border_radius.all(5), padding=5
                )
            ], expand=False
        )

    # --- Methods ---
    def did_mount(self) -> None:
        if self.page:
             self.page.overlay.append(self.file_picker)
             self.page.overlay.append(self.directory_picker)
             self.page.update()
        self.update_view_state()

    def update_view_state(self) -> None:
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
            display_status = self._translate_dynamic_status(status)

            if self.status_text_ref.current: self.status_text_ref.current.value = display_status
            if self.progress_bar_ref.current:
                 self.progress_bar_ref.current.visible = is_running
                 self.progress_bar_ref.current.value = progress if progress is not None else None

            # --- Enable/Disable Controls & Update Button Text ---
            if self.start_button_ref.current:
                self.start_button_ref.current.disabled = is_running
                self.start_button_ref.current.text = self.tr("stop_button") if is_running else self.tr("start_button")
                self.start_button_ref.current.icon = ft.icons.STOP if is_running else ft.icons.PLAY_ARROW
                self.start_button_ref.current.tooltip = self.tr("stop_button_tooltip") if is_running else self.tr("start_button_tooltip")
                self.start_button_ref.current.on_click = self._stop_analysis_clicked if is_running else self._start_analysis_clicked

            results_exist = bool(self.detected_segments)
            if self.save_button_ref.current: self.save_button_ref.current.disabled = not self.enable_controls or not results_exist

            # Enable copy button only if analysis is not running AND songs were identified
            if self.copy_comment_button_ref.current:
                 self.copy_comment_button_ref.current.disabled = is_running or not bool(self.identified_songs)

            # Disable config inputs while running
            self.set_disabled(self.config_refs, is_running)

            # --- Log/Results ---
            if self.log_view_ref.current:
                log_messages = self.view_model.get_log_messages()
                self.log_view_ref.current.controls = [
                    ft.Text(self._translate_dynamic_log(msg), size=12) for msg in log_messages
                ]

            # --- Update Tables ---
            if self.segments_table_ref.current:
                self.segments_table_ref.current.rows = [
                    ft.DataRow(cells=[
                        ft.DataCell(ft.Text(str(i + 1))),
                        ft.DataCell(ft.Text(f"{seg.start:.2f}")),
                        ft.DataCell(ft.Text(f"{seg.end:.2f}")),
                        ft.DataCell(ft.Text(f"{seg.duration:.2f}")),
                    ])
                    for i, seg in enumerate(self.detected_segments)
                ]

            if self.songs_table_ref.current:
                 self.songs_table_ref.current.rows = [
                     ft.DataRow(cells=[
                         ft.DataCell(ft.Text(str(i + 1))),
                         ft.DataCell(ft.Text(song_data.get("title") or self.tr("unknown_title_placeholder"))),
                         ft.DataCell(ft.Text(song_data.get("artist") or self.tr("unknown_artist_placeholder"))),
                         ft.DataCell(ft.Text(str(song_data.get("confidence", "-")).capitalize())),
                     ])
                     for i, song_data in enumerate(self.identified_songs)
                 ]

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
        match_elapsed = re.search(r"\(Elapsed: (\d{2,}:\d{2}:\d{2}|\d{1,2}:\d{2})\)", status)
        if match_elapsed:
            status = status[:match_elapsed.start()].strip()
        loading_prefix = "Loading audio source:"
        if status.startswith(loading_prefix):
            source = status[len(loading_prefix):].strip()
            display_status = f"{self.tr('status_loading_audio')} {source}"
        else:
            translation_key = STATUS_KEY_MAP.get(status)
            display_status = self.tr(translation_key) if translation_key else status
        return display_status

    def _translate_dynamic_log(self, msg: str) -> str:
        """Translates log messages, similar to status."""
        prefix_match = re.match(r"^\[([A-Z]+)\]\s*(.*)", msg)
        if prefix_match:
            core_message = prefix_match.group(2)
            translation_key = LOG_KEY_MAP.get(core_message)
            if translation_key:
                return self.tr(translation_key)
            else:
                return core_message
        else:
            return msg

    # --- Callbacks & Config ---
    def _start_analysis_clicked(self, e: Any) -> None:
        if self.view_model.is_analysis_running():
             print(self.tr("log_stop_requested"))
             self.view_model.request_stop()
        else:
             config_data = self._get_config_from_ui()
             self.view_model.update_full_config(config_data)
             self.view_model.start_analysis()

    def _stop_analysis_clicked(self, e: Any) -> None:
        print("View: Stop button clicked, requesting stop...")
        if self.view_model:
            self.view_model.request_stop()

    def _save_results_clicked(self, e: Any) -> None:
        self.view_model.save_results()

    def _copy_youtube_comment_clicked(self, e: Any) -> None:
        if not self.page: return
        comment_string = self.view_model.get_youtube_comment_string()
        if comment_string:
            self.page.set_clipboard(comment_string)
            snack_bar_msg = self.tr("comment_copied_success")
            snack_bar_color = ft.colors.GREEN
        else:
            snack_bar_msg = self.tr("comment_copied_empty")
            snack_bar_color = ft.colors.AMBER
        snack_bar = ft.SnackBar(
            content=ft.Text(snack_bar_msg),
            bgcolor=snack_bar_color
        )
        self.page.snack_bar = snack_bar
        self.page.snack_bar.open = True
        self.page.update()

    def _on_pick_file_result(self, e: ft.FilePickerResultEvent) -> None:
        if e.files:
            selected_file = e.files[0].path
            if self.file_path_ref.current:
                self.file_path_ref.current.value = selected_file
                if self.source_type_ref.current: self.source_type_ref.current.value = "file"
                self.page.update()

    def _on_pick_directory_result(self, e: ft.FilePickerResultEvent) -> None:
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
             label_key = ""
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
                 self._show_error_snackbar(self.tr("log_invalid_numeric", value=val_str, default=default, label=self.tr(label_key)))
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

    def _show_error_snackbar(self, message: str) -> None:
        if self.page:
            snack_bar = ft.SnackBar(
                content=ft.Text(message),
                bgcolor=ft.colors.RED_400
            )
            self.page.snack_bar = snack_bar
            self.page.snack_bar.open = True
            self.page.update() 