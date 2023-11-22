: <<'END'
시스템 정보
아래 스크립트 삭제시 JF deploy 실행이 안됩니다.
#JF_DEPLOYMENT_INPUT_DATA_INFO_START
{
    "deployment_input_data_form_list": [
        {
            "method": "POST",
            "location": "file",
            "api_key": "image",
            "value_type": "file",
            "category": "image",
            "category_description": "png \uc0ac\uc9c4"
        },
        {
            "method": "POST",
            "location": "form",
            "api_key": "height",
            "value_type": "int",
            "category": "text",
            "category_description": "\ud0a4"
        }
    ]
}
#JF_DEPLOYMENT_INPUT_DATA_INFO_END
END


#!/bin/bash


source activate icon
python FB_apps.py