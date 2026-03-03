from fpdf import FPDF, HTMLMixin
import markdown
import os

class SLMGuide(FPDF, HTMLMixin):
    def header(self):
        if self.page_no() > 1:
            self.set_font('helvetica', 'I', 8)
            self.cell(0, 10, "SLM Platform: End-User Step-by-Step Guide", align='R', new_x="RIGHT", new_y="TOP")
            self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('helvetica', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def create_guide():
    pdf = SLMGuide()
    
    # Cover Page
    pdf.add_page()
    cover_image = r"C:\Users\Administrator\.gemini\antigravity\brain\8f335972-bf27-4b45-beb5-aac29c3c8011\slm_product_guide_cover_1772563179261.png"
    if os.path.exists(cover_image):
        pdf.image(cover_image, x=0, y=0, w=210, h=297)
    
    pdf.add_page()
    
    md_path = r"C:\Users\Administrator\.gemini\antigravity\brain\8f335972-bf27-4b45-beb5-aac29c3c8011\product_guide.md"
    with open(md_path, 'r', encoding='utf-8') as f:
        md_text = f.read()

    # Clean text to avoid unicode errors in standard helvetica
    md_text = md_text.replace('’', "'").replace('“', '"').replace('”', '"').replace('—', '--')
    
    html = markdown.markdown(md_text)
    
    pdf.set_font('helvetica', '', 11)
    pdf.write_html(html)

    output_path = r"C:\Users\Administrator\.gemini\antigravity\brain\8f335972-bf27-4b45-beb5-aac29c3c8011\slm_product_guide.pdf"
    pdf.output(output_path)
    print(f"PDF generated: {output_path}")

if __name__ == "__main__":
    create_guide()
