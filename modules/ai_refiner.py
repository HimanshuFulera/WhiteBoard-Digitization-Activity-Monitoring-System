from google import genai
import os

# You should set this in an environment variable or config, but we'll use the hardcoded one for now
API_KEY = "AIzaSyB7j27vfnnfH0IJT2CWIIjN5b8jOQLJhbo"

def refine_notes_with_gemini(raw_notes_path, refined_notes_path):
    """
    Reads raw OCR notes, sends them to Gemini to clean up overlapping/repeating content,
    and writes the refined notes to a new file.
    """
    if not os.path.exists(raw_notes_path):
        print(f"Error: Raw notes file not found at {raw_notes_path}")
        return False

    with open(raw_notes_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()

    prompt = f"""
You are an expert at cleaning up OCR text extracted from a lecture whiteboard video.
The following text contains timestamps and OCR extractions. Because the teacher writes incrementally, many sections are just slightly longer versions of previous sections. There are also OCR errors.

Your task is to:
1. Merge all the overlapping, incremental notes into a single, cohesive, finalized set of lecture notes.
2. Fix obvious OCR typos and formatting issues to make it readable.
3. Organize the final notes clearly using bullet points and numbered lists.
4. Remove the timestamp markers and section headers from the final output; just provide the clean notes.
5. IMPORTANT: Do NOT use excessive markdown formatting like excessive asterisks (**). Keep the formatting clean and use capital letters for headers rather than bolding everything.

Here is the raw OCR text:

{raw_text}
"""
    print("[Gemini] Sending raw notes to Gemini for refinement...")
    try:
        client = genai.Client(api_key=API_KEY)
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        refined_text = response.text

        with open(refined_notes_path, 'w', encoding='utf-8') as f:
             f.write("========================================================\n")
             f.write("             REFINED LECTURE NOTES (AI)\n")
             f.write("========================================================\n\n")
             f.write(refined_text)

        print(f"[Gemini] Refined notes saved to {refined_notes_path}")
        return True
    except Exception as e:
        print(f"[Gemini] Error during API call: {e}")
        return False
