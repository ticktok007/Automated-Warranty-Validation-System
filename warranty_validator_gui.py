import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import requests
from pathlib import Path
from datetime import datetime
import threading



API_URL = "http://localhost:8080"  # Your API endpoint


class WarrantyValidatorApp:
    
    def __init__(self, root):
        self.root = root
        self.root.title("Warranty Validator - AI Powered System")
        self.root.geometry("1400x850")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.current_image_path = None
        self.current_image = None
        self.validation_history = []
        self.api_connected = False
        
        # Check API connection
        self.check_api_connection()
        
        # Create UI
        self.create_ui()
        
        # Center window
        self.center_window()
    
    def center_window(self):
        """Center window on screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def check_api_connection(self):
        """Check if API is accessible"""
        try:
            response = requests.get(f"{API_URL}/", timeout=2)
            if response.status_code == 200:
                self.api_connected = True
                print("✅ Connected to API")
            else:
                self.api_connected = False
        except:
            self.api_connected = False
            messagebox.showerror(
                "API Connection Error",
                f"Cannot connect to API at {API_URL}\n\n"
                "Please make sure warranty_check.py is running:\n"
                "python warranty_check.py\n\n"
                "The app will continue but validation won't work."
            )
    
    # ══════════════════════════════════════════════════════════════════════════
    # UI CREATION
    # ══════════════════════════════════════════════════════════════════════════
    
    def create_ui(self):
        """Create main UI"""
        
        # Header
        self.create_header()
        
        # Main content (3 columns)
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Configure grid
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=2)
        main_frame.grid_columnconfigure(2, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)
        
        # Left: Image upload
        self.create_left_panel(main_frame)
        
        # Middle: Results
        self.create_middle_panel(main_frame)
        
        # Right: History
        self.create_right_panel(main_frame)
        
        # Footer
        self.create_footer()
    
    def create_header(self):
        """Create header with title and status"""
        
        header_frame = tk.Frame(self.root, bg='#2d3748', height=80)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            header_frame,
            text="🔍 Warranty Validator",
            font=('Arial', 24, 'bold'),
            bg='#2d3748',
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=20)
        
        # Status
        status_text = "✅ API Connected" if self.api_connected else "❌ API Disconnected"
        status_color = "#48bb78" if self.api_connected else "#f56565"
        
        self.status_label = tk.Label(
            header_frame,
            text=status_text,
            font=('Arial', 12),
            bg='#2d3748',
            fg=status_color
        )
        self.status_label.pack(side=tk.RIGHT, padx=20, pady=20)
    
    def create_left_panel(self, parent):
        """Create left panel for image upload"""
        
        left_frame = tk.LabelFrame(
            parent,
            text="📤 Upload Warranty Card",
            font=('Arial', 14, 'bold'),
            bg='white',
            relief=tk.RIDGE,
            borderwidth=2
        )
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        
        # Image display
        self.image_label = tk.Label(
            left_frame,
            text="No image selected\n\nClick 'Browse' to upload",
            bg='#f7fafc',
            width=40,
            height=20,
            relief=tk.SUNKEN,
            borderwidth=2
        )
        self.image_label.pack(padx=20, pady=20)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='white')
        button_frame.pack(pady=10)
        
        browse_btn = tk.Button(
            button_frame,
            text="📁 Browse Image",
            command=self.browse_image,
            font=('Arial', 12, 'bold'),
            bg='#4299e1',
            fg='white',
            width=15,
            height=2,
            cursor='hand2',
            relief=tk.RAISED
        )
        browse_btn.grid(row=0, column=0, padx=5, pady=5)
        
        self.validate_btn = tk.Button(
            button_frame,
            text="✓ Validate",
            command=self.validate_warranty,
            font=('Arial', 12, 'bold'),
            bg='#48bb78',
            fg='white',
            width=15,
            height=2,
            cursor='hand2',
            relief=tk.RAISED,
            state=tk.DISABLED
        )
        self.validate_btn.grid(row=0, column=1, padx=5, pady=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="🗑️ Clear",
            command=self.clear_image,
            font=('Arial', 12, 'bold'),
            bg='#f56565',
            fg='white',
            width=32,
            height=2,
            cursor='hand2',
            relief=tk.RAISED
        )
        clear_btn.grid(row=1, column=0, columnspan=2, pady=5)
    
    def create_middle_panel(self, parent):
        """Create middle panel for results"""
        
        middle_frame = tk.LabelFrame(
            parent,
            text="📊 Validation Results",
            font=('Arial', 14, 'bold'),
            bg='white',
            relief=tk.RIDGE,
            borderwidth=2
        )
        middle_frame.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')
        
        # Progress
        self.progress_frame = tk.Frame(middle_frame, bg='white')
        self.progress_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.progress_label = tk.Label(
            self.progress_frame,
            text="",
            font=('Arial', 10),
            bg='white'
        )
        self.progress_label.pack()
        
        self.progress = ttk.Progressbar(
            self.progress_frame,
            mode='indeterminate',
            length=400
        )
        
        # Results container
        results_container = tk.Frame(middle_frame, bg='white')
        results_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # ML Classification
        ml_frame = tk.LabelFrame(
            results_container,
            text="🤖 ML Classification",
            font=('Arial', 11, 'bold'),
            bg='white'
        )
        ml_frame.pack(fill=tk.X, pady=5)
        
        self.ml_result_label = tk.Label(
            ml_frame,
            text="Waiting for validation...",
            font=('Arial', 20, 'bold'),
            bg='white',
            fg='#718096'
        )
        self.ml_result_label.pack(pady=10)
        
        self.ml_confidence_label = tk.Label(
            ml_frame,
            text="",
            font=('Arial', 12),
            bg='white',
            fg='#4a5568'
        )
        self.ml_confidence_label.pack(pady=5)
        
        # Database Validation
        db_frame = tk.LabelFrame(
            results_container,
            text="💾 Database & OCR Results",
            font=('Arial', 11, 'bold'),
            bg='white'
        )
        db_frame.pack(fill=tk.X, pady=5)
        
        self.db_result_text = scrolledtext.ScrolledText(
            db_frame,
            height=8,
            font=('Courier', 10),
            bg='#f7fafc',
            relief=tk.SUNKEN
        )
        self.db_result_text.pack(padx=10, pady=10, fill=tk.BOTH)
        
        # Final Verdict
        verdict_frame = tk.LabelFrame(
            results_container,
            text="⚖️ Final Verdict",
            font=('Arial', 11, 'bold'),
            bg='white'
        )
        verdict_frame.pack(fill=tk.X, pady=5)
        
        self.verdict_label = tk.Label(
            verdict_frame,
            text="",
            font=('Arial', 18, 'bold'),
            bg='white',
            height=2
        )
        self.verdict_label.pack(pady=10)
        
        self.message_label = tk.Label(
            verdict_frame,
            text="",
            font=('Arial', 10),
            bg='white',
            fg='#4a5568',
            wraplength=500
        )
        self.message_label.pack(pady=5)
    
    def create_right_panel(self, parent):
        """Create right panel for history"""
        
        right_frame = tk.LabelFrame(
            parent,
            text="📜 Validation History",
            font=('Arial', 14, 'bold'),
            bg='white',
            relief=tk.RIDGE,
            borderwidth=2
        )
        right_frame.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')
        
        # History listbox
        self.history_listbox = tk.Listbox(
            right_frame,
            font=('Courier', 9),
            bg='#f7fafc',
            selectmode=tk.SINGLE
        )
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Scrollbar
        scrollbar = tk.Scrollbar(self.history_listbox)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.history_listbox.yview)
        
        # Clear button
        clear_history_btn = tk.Button(
            right_frame,
            text="Clear History",
            command=self.clear_history,
            font=('Arial', 10),
            bg='#ed8936',
            fg='white',
            cursor='hand2'
        )
        clear_history_btn.pack(pady=10)
    
    def create_footer(self):
        """Create footer with stats"""
        
        footer_frame = tk.Frame(self.root, bg='#2d3748', height=40)
        footer_frame.pack(fill=tk.X, side=tk.BOTTOM)
        footer_frame.pack_propagate(False)
        
        self.footer_label = tk.Label(
            footer_frame,
            text=f"API: {API_URL} | Total Validations: 0",
            font=('Arial', 10),
            bg='#2d3748',
            fg='white'
        )
        self.footer_label.pack(pady=10)
    
    # ══════════════════════════════════════════════════════════════════════════
    # FUNCTIONALITY
    # ══════════════════════════════════════════════════════════════════════════
    
    def browse_image(self):
        """Browse and load image"""
        
        file_path = filedialog.askopenfilename(
            title="Select Warranty Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, file_path):
        """Load and display image"""
        
        try:
            self.current_image_path = file_path
            
            # Load image
            img = Image.open(file_path)
            
            # Resize to fit
            img.thumbnail((400, 400))
            
            # Convert to PhotoImage
            self.current_image = ImageTk.PhotoImage(img)
            
            # Display
            self.image_label.configure(image=self.current_image, text="")
            
            # Enable validate button
            self.validate_btn.configure(state=tk.NORMAL)
            
            # Clear previous results
            self.clear_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")
    
    def validate_warranty(self):
        """Validate warranty via API"""
        
        if not self.current_image_path:
            messagebox.showwarning("Warning", "Please select an image first")
            return
        
        if not self.api_connected:
            retry = messagebox.askyesno(
                "API Disconnected",
                f"Cannot connect to API at {API_URL}\n\n"
                "Make sure warranty_check.py is running.\n\n"
                "Retry connection?"
            )
            if retry:
                self.check_api_connection()
                if not self.api_connected:
                    return
            else:
                return
        
        # Disable button
        self.validate_btn.configure(state=tk.DISABLED)
        
        # Show progress
        self.progress_label.configure(text="🔄 Validating warranty card...")
        self.progress.pack()
        self.progress.start(10)
        
        # Run in thread
        thread = threading.Thread(target=self._run_validation)
        thread.daemon = True
        thread.start()
    
    def _run_validation(self):
        """Run validation via API (in separate thread)"""
        
        try:
            # Read image file
            with open(self.current_image_path, 'rb') as f:
                files = {'file': (Path(self.current_image_path).name, f, 'image/jpeg')}
                
                # Call API
                response = requests.post(
                    f"{API_URL}/api/v1/warranty/validate",
                    files=files,
                    timeout=30
                )
            
            if response.status_code == 200:
                result = response.json()
                self.root.after(0, self._display_results, result)
            else:
                error_msg = f"API Error: {response.status_code}\n{response.text[:200]}"
                self.root.after(0, lambda: messagebox.showerror("Validation Error", error_msg))
                self.root.after(0, self._stop_progress)
                
        except requests.exceptions.Timeout:
            self.root.after(0, lambda: messagebox.showerror(
                "Timeout", "Request timed out. Please try again."
            ))
            self.root.after(0, self._stop_progress)
        except requests.exceptions.ConnectionError:
            self.root.after(0, lambda: messagebox.showerror(
                "Connection Error",
                f"Lost connection to API at {API_URL}\n\n"
                "Make sure warranty_check.py is still running."
            ))
            self.root.after(0, self._stop_progress)
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror(
                "Error", f"Validation failed:\n{type(e).__name__}: {e}"
            ))
            self.root.after(0, self._stop_progress)
    
    def _display_results(self, result):
        """Display validation results"""
        
        # Stop progress
        self._stop_progress()
        
        # ML Classification
        if 'ml_classification' in result:
            ml = result['ml_classification']
            prediction = ml.get('prediction', 'unknown').upper()
            confidence = ml.get('confidence', 0) * 100
            
            if prediction == 'AUTHENTIC':
                self.ml_result_label.configure(text="✅ AUTHENTIC", fg='#48bb78')
            else:
                self.ml_result_label.configure(text="❌ FRAUDULENT", fg='#f56565')
            
            probs = ml.get('probabilities', {})
            self.ml_confidence_label.configure(
                text=f"Confidence: {confidence:.2f}%\n"
                     f"Authentic: {probs.get('authentic', 0)*100:.2f}% | "
                     f"Fraudulent: {probs.get('fraudulent', 0)*100:.2f}%"
            )
        
        # Database & OCR results
        self.db_result_text.delete(1.0, tk.END)
        
        db_text = "Validation Details:\n"
        db_text += "━" * 50 + "\n\n"
        
        # Extracted data
        if 'extracted_data' in result and result['extracted_data']:
            db_text += "📄 Extracted Data:\n"
            for key, value in result['extracted_data'].items():
                if value:
                    db_text += f"   {key}: {value}\n"
            db_text += "\n"
        
        # Warranty number
        if 'warranty_number' in result and result['warranty_number']:
            db_text += f"💾 Warranty Number: {result['warranty_number']}\n\n"
        
        # Fraud score
        if 'fraud_risk_score' in result:
            score = result['fraud_risk_score']
            db_text += f"🔍 Fraud Risk Score: {score:.3f}\n"
            if score < 0.3:
                db_text += "   Risk Level: LOW ✅\n"
            elif score < 0.6:
                db_text += "   Risk Level: MEDIUM ⚠️\n"
            else:
                db_text += "   Risk Level: HIGH ❌\n"
        
        db_text += "\n" + "━" * 50
        
        self.db_result_text.insert(1.0, db_text)
        
        # Final Verdict
        verdict = result.get('final_verdict', 'UNKNOWN')
        message = result.get('message', '')
        
        if verdict == 'AUTHENTIC':
            self.verdict_label.configure(
                text="✅ AUTHENTIC WARRANTY",
                fg='white',
                bg='#48bb78'
            )
        elif verdict == 'FRAUDULENT':
            self.verdict_label.configure(
                text="❌ FRAUDULENT WARRANTY",
                fg='white',
                bg='#f56565'
            )
        else:
            self.verdict_label.configure(
                text="⚠️ SUSPICIOUS - REVIEW NEEDED",
                fg='white',
                bg='#ed8936'
            )
        
        self.message_label.configure(text=message)
        
        # Add to history
        self.add_to_history(result)
        
        # Re-enable button
        self.validate_btn.configure(state=tk.NORMAL)
    
    def _stop_progress(self):
        """Stop progress bar"""
        self.progress.stop()
        self.progress.pack_forget()
        self.progress_label.configure(text="")
    
    def add_to_history(self, result):
        """Add validation to history"""
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        verdict = result.get('final_verdict', 'UNKNOWN')
        confidence = result.get('ml_classification', {}).get('confidence', 0) * 100
        filename = Path(self.current_image_path).name[:20]
        
        entry = f"[{timestamp}] {filename:20s} → {verdict:12s} ({confidence:.1f}%)"
        
        self.history_listbox.insert(0, entry)
        self.validation_history.append(result)
        
        # Update footer
        self.footer_label.configure(
            text=f"API: {API_URL} | Total Validations: {len(self.validation_history)}"
        )
    
    def clear_results(self):
        """Clear result displays"""
        self.ml_result_label.configure(text="Waiting for validation...", fg='#718096')
        self.ml_confidence_label.configure(text="")
        self.db_result_text.delete(1.0, tk.END)
        self.verdict_label.configure(text="", bg='white')
        self.message_label.configure(text="")
    
    def clear_image(self):
        """Clear current image"""
        self.current_image_path = None
        self.current_image = None
        
        self.image_label.configure(
            image='',
            text="No image selected\n\nClick 'Browse' to upload"
        )
        
        self.validate_btn.configure(state=tk.DISABLED)
        self.clear_results()
    
    def clear_history(self):
        """Clear validation history"""
        if messagebox.askyesno("Confirm", "Clear all validation history?"):
            self.history_listbox.delete(0, tk.END)
            self.validation_history.clear()
            
            self.footer_label.configure(
                text=f"API: {API_URL} | Total Validations: 0"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app = WarrantyValidatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
