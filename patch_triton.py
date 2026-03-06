import os
import site

def patch_triton():
    try:
        site_packages = site.getsitepackages()[0]
        triton_compiler_path = os.path.join(site_packages, "triton", "compiler", "compiler.py")
        triton_backends_compiler_path = os.path.join(site_packages, "triton", "backends", "compiler.py")
        
        patch_code = "\n\n# --- PATCH FOR VLLM WINDOWS COMPATIBILITY ---\nclass AttrsDescriptor:\n    def __init__(self, *args, **kwargs):\n        pass\n# ----------------------------------------------\n"
        
        patched_count = 0
        
        for path in [triton_compiler_path, triton_backends_compiler_path]:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                
                if "class AttrsDescriptor" not in content:
                    with open(path, "a", encoding="utf-8") as f:
                        f.write(patch_code)
                    print(f"Patched: {path}")
                    patched_count += 1
                else:
                    print(f"Already patched: {path}")
            else:
                print(f"File not found: {path}. Creating it with minimal definition.")
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(patch_code)
                    print(f"Created: {path}")
                    patched_count += 1
                except Exception as inner_e:
                    print(f"Could not create file {path}: {inner_e}")
                
        if patched_count > 0:
            print(f"Successfully applied {patched_count} patch(es) to triton-windows.")
        else:
            print("No action needed or paths not found.")
            
    except Exception as e:
        print(f"Failed to patch triton: {e}")

if __name__ == "__main__":
    patch_triton()
