import base64

def icon_to_base64(icon_path):
    with open(icon_path, "rb") as icon_file:
        icon_data = icon_file.read()
        return base64.b64encode(icon_data).decode('utf-8')

def load_css():
    base64_icon_001 = icon_to_base64("utils/static/reload_icon_narrow_sml.png")
    base64_icon_002 = icon_to_base64("utils/static/scanner_icon_narrow_sml.png")

    css = f"""
    #upload_btn {{
        background-image: url('data:image/x-icon;base64,{base64_icon_001}');
        background-repeat: no-repeat;
        background-position: center;
        background-size: full;
        margin: 0em 0em 0em 0;
        max-width: 1.85em;
        min-width: 1.85em !important;
        height: 5.85em;
        padding: 0em 0em 0em 0em;
        border: none;
    }}
    
    #scan_btn {{
        background-image: url('data:image/x-icon;base64,{base64_icon_002}');
        background-repeat: no-repeat;
        background-position: center;
        background-size: full;
        margin: 0em 0em 0em 0;
        max-width: 1.85em;
        min-width: 1.85em !important;
        height: 5.85em;
        padding: 0em 0em 0em 0em;
        border: none;
    }}

    input[type='number'] {{
        text-align: center;
        -moz-appearance: textfield;
    }}

    input[type='number']::-webkit-outer-spin-button,
    input[type='number']::-webkit-inner-spin-button {{
        -webkit-appearance: none;
        margin: 0;
    }}

    input[type='number'] {{
        appearance: none;
        -moz-appearance: textfield;
    }}

    #image-gallery {{
        max-height: 585px !important;
        height: auto !important;
    }}

    #output_caption_textbox {{
        height: 434px !important;
        overflow-y: auto !important;
        resize: none !important;
    }}

    #output_prompt_textbox textarea {{
        height: 6em !important;
        max-height: 6em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}

    #output_prompt_textbox_batch textarea {{
        height: 3em !important;
        max-height: 3em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}

    #folder_input_textbox textarea {{
        height: 3em !important;
        max-height: 3em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}
    
    #folder_input_output_textbox textarea {{
        height: 4.54em !important;
        max-height: 4.54em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}
    
    #model_folder_input_textbox textarea {{
        height: 3.58em !important;
        max-height: 3.58em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}
    
    #model_folder_input_textbox_tall textarea {{
        height: 4.94em !important;
        max-height: 4.94em !important;
        overflow-y: auto !important;
        resize: none !important;
    }}
    
    #scan_summary {{
        height: 260px !important;  /* Approximately 5 lines of text */
        overflow-y: auto !important;
        resize: none !important;
    }}

    #prompt_input_display_lock {{
        height: 206px !important;
        overflow-y: auto !important;
        resize: none !important;
    }}

    #edit_caption_display_lock {{
        height: 190px !important;
        overflow-y: auto !important;
        resize: none !important;
    }}

    .gallery-scroll {{
        overflow-y: scroll !important;
        height: 580px;
    }}

    * {{
        scrollbar-width: none;
        -ms-overflow-style: none;
    }}

    *::-webkit-scrollbar {{
        width: 0px;
        background: transparent;
    }}

    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

    body, input, button, textarea {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    #highlighted-text {{
        font-family: 'JetBrains Mono', monospace !important;
    }}

    #image-gallery .label, #image .label {{
        display: none !important;
    }}

    #permanent_image .icon-buttons.svelte-1p15vfy {{
        display: none !important;
    }}

    #extra_options .selectbox__value__label {{
        display: none !important;
    }}

    #permanent_image img {{
        width: 100% !important;
        height: 1176px;
        object-fit: cover !important;
        border: 2px solid #9b30ff !important;
        border-radius: 5px;
        box-sizing: border-box;
    }}

    #generate_btn {{
        color: #ff7900 !important;
        background-color: #9b30ff !important;
        border: none !important;
    }}

    #generate_btn:hover {{
        color: #9b30ff !important;
        background-color: #ff7900 !important;
    }}

    #inverse_generate_tall_btn {{
        height: 3.5em !important;
        border: none !important;
    }}

    #inverse_generate_tall_btn:hover {{
        color: #9b30ff !important;
        background-color: #ff7900 !important;
    }}

    #inverse_generate_btn {{
        border: none !important;
    }}

    #inverse_generate_btn:hover {{
        color: #9b30ff !important;
        background-color: #ff7900 !important;
    }}
    
    #upload_file_area {{
        height: 5em !important;
    }}
    
    <style>
      .svelte-1rjryqp {{
        display: none !important;
      }}
      .show-api.svelte-1rjryqp, footer.svelte-1rjryqp {{
        display: none !important;
      }}
      .tabs button.selected {{
        border: 2px solid #5DADE2 !important;
        border-radius: 5px !important;
      }}
      .tabs button, .iconify.iconify--mdi, .icon.svelte-1oiin9d, .dropdown-arrow.svelte-xjn76a {{
        color: #ff8c00 !important;
      }}
      .icon.svelte-snayfm.selected {{
        border: 2px solid #ff8c00 !important;
        border-radius: 5px !important;
      }}
    </style>
    """

    css += """
    <script src="https://unpkg.com/react@17/umd/react.development.js"></script>
    <script src="https://unpkg.com/react-dom@17/umd/react-dom.development.js"></script>
    <script src="https://unpkg.com/recharts@2.12.0/umd/Recharts.js"></script>
    <script type="module">
        document.addEventListener('DOMContentLoaded', () => {
            const container = document.getElementById('training-visualizer');
            if (container) {
                import('/static/js/training-visualizer.js').then(module => {
                    const root = ReactDOM.createRoot(container);
                    root.render(React.createElement(module.default));
                }).catch(err => console.error('Error loading training visualizer:', err));
            }
        });
    </script>
    """
    
    return css
