"""
app_view.py
------------
Main Flet GUI view for the Singing Analysis application.
Displays UI controls, binds to the AnalysisViewModel, and handles user interaction.
All translation key maps are module-level constants for maintainability.
"""
import re
from typing import Any, Callable, Dict, List
from datetime import datetime, timedelta

import flet as ft

from viewmodel.analysis_viewmodel import AnalysisViewModel

# --- Time String Utility Functions ---
def time_str_to_seconds(time_str: str) -> float:
    """
    Converts a time string (hh:mm:ss.ms, mm:ss.ms, ss.ms) to seconds.
    Returns 0.0 on error.
    """
    if not time_str or not isinstance(time_str, str):
        return 0.0
    parts = time_str.split(':')
    seconds = 0.0
    try:
        if len(parts) == 3: # hh:mm:ss.ms
            seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2: # mm:ss.ms
            seconds = int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1: # ss.ms
            seconds = float(parts[0])
        else:
             return 0.0 # Invalid format
    except ValueError:
        return 0.0 # Parsing error
    return seconds

def format_seconds_to_hms(total_seconds: float) -> str:
    """
    Formats total seconds into hh:mm:ss.ms string.
    Ensures three decimal places for milliseconds.
    Returns '0:00.000' if input is invalid or None.
    """
    if total_seconds is None or total_seconds < 0:
        return "0:00.000"
    
    try:
        # Use timedelta for easier calculation
        td = timedelta(seconds=total_seconds)
        
        # Extract components
        total_secs_int = int(td.total_seconds())
        milliseconds = int((td.total_seconds() - total_secs_int) * 1000)
        
        hours, remainder = divmod(total_secs_int, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
             return f"{hours}:{minutes:02}:{seconds:02}.{milliseconds:03}"
        else:
             return f"{minutes:02}:{seconds:02}.{milliseconds:03}"
    except Exception:
        return "0:00.000" # Fallback

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
            "source_type", "file_path", "url", "output_dir", "sing_ref_time", "non_sing_ref_time",
            "hmm_threshold",
            "min_duration_for_id",
            "whisper_model", "enable_hpss", "api_key",
            "analysis_window_seconds",
            "save_json_checkbox", "save_csv_checkbox", "save_youtube_comment_checkbox",
            "config_panel", "browse_file_button", "select_dir_button", "copy_comment_button",
            "ffmpeg_path", "check_ffmpeg_button", "elapsed_time"
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
            self.file_path_ref, self.url_ref, self.output_dir_ref,
            self.sing_ref_time_ref, self.non_sing_ref_time_ref,
            self.analysis_window_seconds_ref,
            self.hmm_threshold_ref,
            self.enable_hpss_ref,
            self.min_duration_for_id_ref,
            self.whisper_model_ref, self.api_key_ref,
            self.ffmpeg_path_ref,
            self.save_json_checkbox_ref, self.save_csv_checkbox_ref, self.save_youtube_comment_checkbox_ref
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
        # --- make_config_field Helper (Updated) ---
        def make_config_field(ref, config_key, label_key, col_spans: Dict[str, int], keyboard_type=ft.KeyboardType.TEXT, is_dropdown=False, dropdown_options=None, is_checkbox=False, is_password=False, expand=False, is_timestamp=False):
            common_tooltip = self.tr(label_key + "_tooltip", default="") # Use default if tooltip missing
            control = None # Initialize control
            initial_value = self.view_model.get_config_value(config_key)
            
            # Format initial value if it's a timestamp
            display_value = str(initial_value)
            if is_timestamp:
                display_value = format_seconds_to_hms(initial_value)
                keyboard_type = ft.KeyboardType.TEXT # Timestamps are now strings like hh:mm:ss

            if is_dropdown:
                control = ft.Dropdown(
                    ref=ref,
                    label=self.tr(label_key),
                    value=initial_value, # Use raw value for dropdown
                    options=[ft.dropdown.Option(opt) for opt in dropdown_options],
                    tooltip=common_tooltip,
                    expand=expand, # Dropdown can expand horizontally
                    dense=True
                )
            elif is_checkbox:
                control = ft.Checkbox(
                    ref=ref,
                    label=self.tr(label_key),
                    value=initial_value, # Use raw boolean value
                    tooltip=common_tooltip
                )
            else:
                control = ft.TextField(
                    ref=ref,
                    label=self.tr(label_key),
                    value=display_value, # Use potentially formatted value
                    hint_text=self.tr(label_key + "_hint", default="hh:mm:ss.ms" if is_timestamp else None), # Add specific hint for timestamps
                    keyboard_type=keyboard_type,
                    password=is_password,
                    can_reveal_password=is_password,
                    expand=expand,
                    tooltip=common_tooltip,
                    height=45,
                    dense=True
                )
            # Wrap the control in a Container and apply column spans
            return ft.Container(content=control, padding=ft.padding.only(right=5, bottom=5), col=col_spans)

        # --- Quick Config Guide ---
        guide_text = ft.Container(
            content=ft.Text(
                self.tr("help_text"),
                size=11,
                style=ft.TextStyle(italic=False),
                text_align=ft.TextAlign.LEFT,
                color=ft.colors.WHITE
            ),
            margin=ft.margin.only(bottom=10),
            padding=8,
            border_radius=5,
            bgcolor=ft.colors.with_opacity(0.2, ft.colors.BLUE_GREY_700),
            width=float("inf"),
            border=ft.border.all(1, ft.colors.BLUE_GREY_400)
        )

        # --- Define Config Controls within Responsive Rows ---
        config_controls = [
            # Add the guide text at the top
            guide_text,
            
            # Row 1: Reference Times & Window
            ft.ResponsiveRow([
                make_config_field(self.sing_ref_time_ref, 'singing_ref_time', 'sing_ref_time_label', col_spans={"sm": 12, "md": 4}, is_timestamp=True),
                make_config_field(self.non_sing_ref_time_ref, 'non_singing_ref_time', 'non_sing_ref_time_label', col_spans={"sm": 12, "md": 4}, is_timestamp=True),
                make_config_field(self.analysis_window_seconds_ref, 'analysis_window_seconds', 'analysis_window_seconds_label', col_spans={"sm": 12, "md": 4}, keyboard_type=ft.KeyboardType.NUMBER),
            ], alignment=ft.MainAxisAlignment.START),

            # Row 2: HMM Threshold and HPSS
            ft.ResponsiveRow([
                make_config_field(self.hmm_threshold_ref, 'hmm_threshold', 'hmm_thresh_label', col_spans={"sm": 6, "md": 6}, keyboard_type=ft.KeyboardType.NUMBER),
                make_config_field(self.enable_hpss_ref, 'enable_hpss', 'enable_hpss_checkbox', col_spans={"sm": 6, "md": 6}, is_checkbox=True),
            ], alignment=ft.MainAxisAlignment.START),

            # Row 3: Identification Duration & Model
            ft.ResponsiveRow([
                make_config_field(self.min_duration_for_id_ref, 'min_duration_for_id', 'min_id_dur_label', col_spans={"sm": 12, "md": 6}, keyboard_type=ft.KeyboardType.NUMBER),
                make_config_field(self.whisper_model_ref, 'whisper_model', 'whisper_model_label', col_spans={"sm": 12, "md": 6}, is_dropdown=True, dropdown_options=["tiny", "base", "small", "medium", "large"]),
            ], alignment=ft.MainAxisAlignment.START),

            # Row 4: API Key
            ft.ResponsiveRow([
                make_config_field(self.api_key_ref, 'gemini_api_key', 'gemini_api_key_label', col_spans=12, is_password=True, expand=True),
            ]),

            # Row 5: Output Directory
            ft.ResponsiveRow([
                 make_config_field(self.output_dir_ref, 'output_dir', 'output_dir_label', col_spans={"sm": 12, "md": 9, "lg": 10}, keyboard_type=ft.KeyboardType.TEXT, expand=True),
                 # Button needs its own container with col spans
                 ft.Container(
                    content=ft.ElevatedButton(
                        self.tr("select_button"), ref=self.select_dir_button_ref,
                        on_click=lambda _: self.directory_picker.get_directory_path(),
                        tooltip=self.tr("select_button_tooltip")
                    ),
                    padding=ft.padding.only(top=5), # Add some padding to align button better
                    col={"sm": 12, "md": 3, "lg": 2}
                 )
            ]),

            # Row 6: FFmpeg Path & Check Button
            ft.ResponsiveRow([
                make_config_field(self.ffmpeg_path_ref, 'ffmpeg_path', 'ffmpeg_path_label', col_spans={"sm": 12, "md": 9, "lg": 10}, keyboard_type=ft.KeyboardType.TEXT, expand=True),
                # Button to check FFmpeg availability
                ft.Container(
                    content=ft.ElevatedButton(
                        self.tr("check_ffmpeg_button"), ref=self.check_ffmpeg_button_ref,
                        on_click=self._check_ffmpeg_clicked,
                        tooltip=self.tr("check_ffmpeg_button_tooltip")
                    ),
                    padding=ft.padding.only(top=5),
                    col={"sm": 12, "md": 3, "lg": 2}
                )
            ]),

            # Row 7: Save Options
            ft.ResponsiveRow([
                make_config_field(self.save_json_checkbox_ref, 'save_results_json', 'save_json_checkbox', col_spans={"sm": 4}, is_checkbox=True),
                make_config_field(self.save_csv_checkbox_ref, 'save_results_dataframe', 'save_csv_checkbox', col_spans={"sm": 4}, is_checkbox=True),
                make_config_field(self.save_youtube_comment_checkbox_ref, 'save_youtube_comment', 'save_youtube_comment_checkbox', col_spans={"sm": 4}, is_checkbox=True),
            ], alignment=ft.MainAxisAlignment.START),
        ]

        return ft.ExpansionPanelList(
            expand_icon_color=ft.colors.WHITE,
            elevation=8,
            divider_color=ft.colors.BLUE_GREY_400,
            controls=[
                ft.ExpansionPanel(
                    ref=self.config_panel_ref,
                    header=ft.ListTile(title=ft.Text(self.tr("config_title"))),
                    content=ft.Container(
                        padding=10,
                        content=ft.Column(config_controls, spacing=5, alignment=ft.MainAxisAlignment.START)
                    ),
                    can_tap_header=True, # Allow tapping header to expand/collapse
                    # Initially expanded - consider view_model state if needed
                    # expanded = self.view_model.get_config_value('config_panel_expanded', True) 
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
                        ft.Row([
                            ft.ProgressBar(ref=self.progress_bar_ref, value=None, width=220, bar_height=10, visible=False),
                            ft.Text(ref=self.elapsed_time_ref, value="", size=14, weight=ft.FontWeight.BOLD, visible=True)
                        ], spacing=10, alignment=ft.MainAxisAlignment.CENTER),
                        ft.Text("", ref=self.status_text_ref, visible=False)  # Keep ref but hide it
                     ], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
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
                         )
                    ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
                ),
                ft.Tab(
                    text=self.tr("identified_songs_tab"), icon=ft.icons.QUEUE_MUSIC,
                     content=ft.Column([
                         ft.Container(
                             content=ft.DataTable(
                                ref=self.songs_table_ref,
                                columns=[
                                    ft.DataColumn(ft.Text(self.tr("song_col_seg_num"))),
                                    ft.DataColumn(ft.Text(self.tr("song_col_title"))),
                                    ft.DataColumn(ft.Text(self.tr("song_col_artist"))),
                                    ft.DataColumn(ft.Text(self.tr("song_col_confidence"))),
                                ], rows=[], column_spacing=20, heading_row_height=35, data_row_max_height=40,
                             ), expand=True
                         )
                     ], scroll=ft.ScrollMode.ADAPTIVE, expand=True)
                ),
                 ft.Tab(
                    text=self.tr("actions_tab"), icon=ft.icons.SETTINGS_APPLICATIONS,
                    content=ft.Container(
                        padding=15,
                        content=ft.Column([
                            ft.Row([
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
                        ])
                    )
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
            
            # Extract elapsed time if present in the status message
            elapsed_time = ""
            match_elapsed = re.search(r"\(Elapsed: (\d{2,}:\d{2}:\d{2}|\d{1,2}:\d{2})\)", status)
            if match_elapsed:
                elapsed_time = match_elapsed.group(1)
            elif is_running and self.view_model._start_time:
                # Calculate elapsed time manually if not in the status message
                elapsed = datetime.now() - self.view_model._start_time
                total_seconds = int(elapsed.total_seconds())
                minutes = total_seconds // 60
                seconds = total_seconds % 60
                elapsed_time = f"{minutes:02}:{seconds:02}"
            
            # Update status text (hidden but kept for backward compatibility)
            if self.status_text_ref.current: 
                self.status_text_ref.current.value = display_status
            
            # Update progress bar and elapsed time
            if self.progress_bar_ref.current:
                self.progress_bar_ref.current.visible = is_running
                self.progress_bar_ref.current.value = progress if progress is not None else None
            
            # Update elapsed time display
            if self.elapsed_time_ref.current:
                if is_running:
                    self.elapsed_time_ref.current.value = f"{self.tr('elapsed_label')}: {elapsed_time}" if elapsed_time else ""
                else:
                    self.elapsed_time_ref.current.value = ""

            # --- Enable/Disable Controls & Update Button Text ---
            if self.start_button_ref.current:
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
                         ft.DataCell(ft.Text(
                             "❌ " + self.tr("identification_failed") if song_data.get("error") 
                             else (song_data.get("title") or self.tr("unknown_title_placeholder"))
                         )),
                         ft.DataCell(ft.Text(
                             song_data.get("error", "")[:50] + "..." if song_data.get("error") and len(song_data.get("error", "")) > 50
                             else (song_data.get("artist") or self.tr("unknown_artist_placeholder"))
                         )),
                         ft.DataCell(ft.Text(
                             self.tr("not_available") if song_data.get("error") 
                             else str(song_data.get("confidence", "-")).capitalize()
                         )),
                     ])
                     for i, song_data in enumerate(self.identified_songs)
                 ]

            # --- Final Page Update ---
            self.page.update()

        if self.page: _update_ui()

    # --- Translation Helpers for Dynamic Content ---
    def _translate_dynamic_status(self, status: str) -> str:
        """Translates status messages that might contain placeholders or numbers."""
        # Extract base status without the elapsed time part for translation purposes
        base_status = status
        match_elapsed = re.search(r"\(Elapsed: (\d{2,}:\d{2}:\d{2}|\d{1,2}:\d{2})\)", status)
        if match_elapsed:
            base_status = status[:match_elapsed.start()].strip()
        
        # Process base status for translation
        loading_prefix = "Loading audio source:"
        if base_status.startswith(loading_prefix):
            source = base_status[len(loading_prefix):].strip()
            display_status = f"{self.tr('status_loading_audio')} {source}"
        else:
            translation_key = STATUS_KEY_MAP.get(base_status)
            display_status = self.tr(translation_key) if translation_key else base_status
            
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
    def _get_numeric_from_ref(self, ref: ft.Ref, default: Any, is_float: bool = True) -> Any:
        """Helper to safely get numeric value from a TextField Ref."""
        if ref.current:
            val_str = ref.current.value
            try:
                return float(val_str) if is_float else int(val_str)
            except (ValueError, TypeError):
                return default
        return default

    def _get_text_from_ref(self, ref: ft.Ref, default: str = '') -> str:
        """Helper to safely get text value from a TextField Ref."""
        return ref.current.value if ref.current and ref.current.value else default

    def _get_bool_from_ref(self, ref: ft.Ref, default: bool = False) -> bool:
        """Helper to safely get boolean value from a Checkbox Ref."""
        return ref.current.value if ref.current else default

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
        if not self.page: return
        
        success = self.view_model.save_results()
        
        # Show appropriate snackbar based on success status
        if success:
            snack_bar = ft.SnackBar(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            name=ft.icons.SAVE_ALT,
                            color=ft.colors.WHITE
                        ),
                        ft.Text(
                            self.tr("save_results_success"),
                            size=16,
                            weight=ft.FontWeight.BOLD
                        )
                    ]),
                    padding=10
                ),
                bgcolor=ft.colors.GREEN_700,
                action="OK",
                action_color=ft.colors.WHITE
            )
        else:
            snack_bar = ft.SnackBar(
                content=ft.Container(
                    content=ft.Row([
                        ft.Icon(
                            name=ft.icons.ERROR_OUTLINE,
                            color=ft.colors.WHITE
                        ),
                        ft.Text(
                            self.tr("save_results_error"),
                            size=16,
                            weight=ft.FontWeight.BOLD
                        )
                    ]),
                    padding=10
                ),
                bgcolor=ft.colors.RED_600,
                action="OK",
                action_color=ft.colors.WHITE
            )
            
        self.page.snack_bar = snack_bar
        self.page.snack_bar.open = True
        self.page.update()

    def _copy_youtube_comment_clicked(self, e: Any) -> None:
        if not self.page: return
        comment_string = self.view_model.get_youtube_comment_string()
        if comment_string:
            self.page.set_clipboard(comment_string)
            snack_bar_msg = self.tr("comment_copied_success")
            snack_bar_color = ft.colors.GREEN_700
            # Add to log messages for better visibility
            self.view_model.log_messages.append(f"[INFO] {snack_bar_msg}")
        else:
            snack_bar_msg = self.tr("comment_copied_empty")
            snack_bar_color = ft.colors.AMBER
        
        # Create a more prominent snackbar
        snack_bar = ft.SnackBar(
            content=ft.Container(
                content=ft.Row([
                    ft.Icon(
                        name=ft.icons.CONTENT_COPY if comment_string else ft.icons.ERROR_OUTLINE,
                        color=ft.colors.WHITE
                    ),
                    ft.Text(
                        snack_bar_msg,
                        size=16,
                        weight=ft.FontWeight.BOLD
                    )
                ]),
                padding=10
            ),
            bgcolor=snack_bar_color,
            action="OK",
            action_color=ft.colors.WHITE
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
        """Gathers all configuration values from the UI controls."""
        config = {}
        config['source_type'] = self.source_type_ref.current.value if self.source_type_ref.current else 'file'
        config['file'] = self._get_text_from_ref(self.file_path_ref) if config['source_type'] == 'file' else ''
        config['url'] = self._get_text_from_ref(self.url_ref) if config['source_type'] == 'url' else ''

        # Handle Timestamps with new utility function
        config['singing_ref_time'] = time_str_to_seconds(self._get_text_from_ref(self.sing_ref_time_ref))
        config['non_singing_ref_time'] = time_str_to_seconds(self._get_text_from_ref(self.non_sing_ref_time_ref))
        
        # Use standard helper for other numeric fields
        config['analysis_window_seconds'] = self._get_numeric_from_ref(self.analysis_window_seconds_ref, 10.0) # Use default from DEFAULT_CONFIG
        config['hmm_threshold'] = self._get_numeric_from_ref(self.hmm_threshold_ref, 0.6) # Use default
        config['min_duration_for_id'] = self._get_numeric_from_ref(self.min_duration_for_id_ref, 5.0) # Use default

        # Use standard helper for boolean/checkbox
        config['enable_hpss'] = self._get_bool_from_ref(self.enable_hpss_ref)

        # Use standard helper for text/dropdown/password
        config['whisper_model'] = self.whisper_model_ref.current.value if self.whisper_model_ref.current else 'base'
        config['gemini_api_key'] = self._get_text_from_ref(self.api_key_ref)
        config['output_dir'] = self._get_text_from_ref(self.output_dir_ref)
        config['ffmpeg_path'] = self._get_text_from_ref(self.ffmpeg_path_ref)
        
        # --- Save Options ---
        config['save_results_json'] = self._get_bool_from_ref(self.save_json_checkbox_ref, default=True)
        config['save_results_dataframe'] = self._get_bool_from_ref(self.save_csv_checkbox_ref, default=True)
        config['save_youtube_comment'] = self._get_bool_from_ref(self.save_youtube_comment_checkbox_ref, default=True)

        # --- Validate Timestamps ---
        # Example: Ensure singing ref time is less than non-singing ref time if both are provided
        # This validation might be better placed in the ViewModel or Pipeline
        # if config['singing_ref_time'] > 0 and config['non_singing_ref_time'] > 0 and config['singing_ref_time'] >= config['non_singing_ref_time']:
        #     self._show_error_snackbar("Singing reference time must be less than non-singing reference time.")
        #     # Optionally return None or raise an error to prevent proceeding
        #     return None # Or raise ValueError("Invalid reference times")

        return config

    def _show_error_snackbar(self, message: str) -> None:
        if not self.page: return
        snack_bar = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=ft.colors.RED_400
        )
        self.page.snack_bar = snack_bar
        self.page.snack_bar.open = True
        self.page.update()

    def _check_ffmpeg_clicked(self, e: Any) -> None:
        """Check if ffmpeg is available at the specified path or in system PATH."""
        from singing_detection.identification.audio_segment_utils import AudioSegmentExtractor
        import os
        
        ffmpeg_path = self._get_text_from_ref(self.ffmpeg_path_ref)
        
        # If empty, try to find ffmpeg automatically
        if not ffmpeg_path:
            temp_extractor = AudioSegmentExtractor("", "")
            ffmpeg_path = temp_extractor.ffmpeg_path
            if ffmpeg_path != "ffmpeg":  # If a path was found (not just the command)
                self.ffmpeg_path_ref.current.value = ffmpeg_path
                self.page.update()
        
        # If the path is a directory, check for ffmpeg.exe in it
        elif os.path.isdir(ffmpeg_path):
            possible_exes = ["ffmpeg.exe", "ffmpeg"]
            for exe in possible_exes:
                full_path = os.path.join(ffmpeg_path, exe)
                if os.path.exists(full_path):
                    # Update to use the executable path, not just the directory
                    ffmpeg_path = full_path
                    self.ffmpeg_path_ref.current.value = ffmpeg_path
                    self.page.update()
                    break
        
        # Now check if ffmpeg is available
        is_available, message = AudioSegmentExtractor.check_ffmpeg_availability(ffmpeg_path)
        
        if is_available:
            self._show_success_snackbar(f"{self.tr('ffmpeg_available_msg')}\n{message}")
        else:
            self._show_error_snackbar(f"{self.tr('ffmpeg_not_available_msg')}\n{message}")

    def _show_success_snackbar(self, message: str) -> None:
        """Shows a success snackbar with the given message."""
        snack = ft.SnackBar(
            content=ft.Text(message),
            bgcolor=ft.colors.GREEN_700,
            action="OK"
        )
        self.page.snack_bar = snack
        self.page.snack_bar.open = True
        self.page.update() 