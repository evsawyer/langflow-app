from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema.message import Message
from langflow.schema.content_block import ContentBlock
from langflow.schema.content_types import MediaContent
import base64
import tempfile
from pathlib import Path
import httpx
import json

import cloudinary
import cloudinary.uploader

class GPTImageTool(Component):
    display_name = "GPT Image Generator/Editor"
    description = "Generate or edit an image using OpenAI's gpt-image-1 model"
    icon = "image"
    name = "GPTImageTool"

    inputs = [
        MessageTextInput(
            name="prompt",
            display_name="Prompt",
            info="Describe the image you want",
            value="A futuristic city on Mars at sunset",
            tool_mode=True,
        ),
        MessageTextInput(
            name="api_key",
            display_name="OpenAI API Key",
            info="Your OpenAI API key with gpt-image-1 access",
            value=""
        ),
        MessageTextInput(
            name="cloudinary_api_key",
            display_name="Cloudinary API Key",
            info="",
            value=""
        ),
        MessageTextInput(
            name="cloudinary_api_secret",
            display_name="Cloudinary API Secret",
            info="",
            value=""
        ),
        MessageTextInput(
            name="cloudinary_cloud_name",
            display_name="Cloudinary Cloud Name",
            info="",
            value=""
        ),
        MessageTextInput(
            name="size",
            display_name="Size",
            info="Image size (1024x1024, 1792x1024, or 1024x1792)",
            value="1024x1024"
        ),
        MessageTextInput(
            name="image_url",
            display_name="Image URL(s) (optional)",
            info="Either a single image URL or a JSON array of image URLs"
        ),
    ]

    outputs = [
        Output(display_name="Image Message", name="output", method="build_output", field_type="Message"),
    ]

    async def get_image_bytes(self) -> str:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            temp_files = []
    
            if self.image_url:
                # Parse image URL(s)
                try:
                    image_urls = json.loads(self.image_url)
                    if not isinstance(image_urls, list):
                        image_urls = [image_urls]
                except json.JSONDecodeError:
                    image_urls = [self.image_url]
    
                # Prepare multipart files
                files = [
                    ("model", (None, "gpt-image-1")),
                    ("prompt", (None, self.prompt)),
                ]
    
                async with httpx.AsyncClient(timeout=30.0) as client:
                    for idx, url in enumerate(image_urls):
                        img_response = await client.get(url)
                        img_response.raise_for_status()
                        temp_path = Path(tempfile.mkstemp(suffix=".png")[1])
                        temp_path.write_bytes(img_response.content)
                        temp_files.append(temp_path)
                        files.append(
                            ("image[]", (f"image{idx}.png", temp_path.open("rb"), "image/png"))
                        )
    
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/images/edits",
                        headers=headers,
                        files=files,
                    )
                    response.raise_for_status()
            else:
                # Image generation mode
                headers["Content-Type"] = "application/json"
                payload = {
                    "model": "gpt-image-1",
                    "prompt": self.prompt,
                    "n": 1,
                    "size": self.size
                }
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        "https://api.openai.com/v1/images/generations",
                        headers=headers,
                        json=payload
                    )
                    response.raise_for_status()
    
            result = response.json()
            image_data = result["data"][0]
    
            if "b64_json" in image_data:
                image_bytes = base64.b64decode(image_data["b64_json"])
            elif "url" in image_data:
                img_resp = await httpx.AsyncClient().get(image_data["url"])
                img_resp.raise_for_status()
                image_bytes = img_resp.content
            else:
                raise ValueError("No image data found in OpenAI response.")
    
            # Clean up temp files
            for f in temp_files:
                try:
                    f.unlink()
                except Exception:
                    pass
    
            return f"data:image/png;base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    
        except Exception as e:
            raise RuntimeError(f"❌ Error in get_image_bytes: {str(e)}")
    
                
            
    async def upload_to_cloudinary(self, data_uri: str) -> str:
        cloudinary.config(
            cloud_name=self.cloudinary_cloud_name,
            api_key=self.cloudinary_api_key,
            api_secret=self.cloudinary_api_secret,
            secure=True
        )
    
        result = cloudinary.uploader.upload(data_uri)
        if "secure_url" not in result:
            raise RuntimeError(f"Cloudinary upload failed: {result}")
            return "no url generated"
    
        return result["secure_url"]

        
    async def build_output(self) -> Message:
        try:
            data_uri = await self.get_image_bytes()
            cloudinary_url = await self.upload_to_cloudinary(data_uri)
    
            media_content = MediaContent(
                type="media",
                urls=[cloudinary_url]
            )
    
            content_block = ContentBlock(
                title="Edited Image" if self.image_url else "Generated Image",
                contents=[media_content]
            )
    
            return Message(
                text=cloudinary_url,
                content_blocks=[content_block],
                sender="AI",
                sender_name="GPT Image Generator"
            )
    
        except Exception as e:
            return Message(
                text=f"❌ Error building output: {str(e)}",
                sender="AI",
                sender_name="GPT Image Generator"
            )

    