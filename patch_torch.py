import os
import site

def patch_torch_inductor():
    try:
        site_packages = site.getsitepackages()[1]
        hints_path = os.path.join(site_packages, "torch", "_inductor", "runtime", "hints.py")
        
        if not os.path.exists(hints_path):
            print(f"File not found: {hints_path}")
            return
            
        with open(hints_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        target_imports = [
            "from triton.backends.compiler import AttrsDescriptor",
            "from triton.compiler.compiler import AttrsDescriptor"
        ]
        
        patched = False
        for target in target_imports:
            if target in content:
                content = content.replace(target, "AttrsDescriptor = object  # PATCHED FOR WINDOWS")
                patched = True
                
        if patched:
            with open(hints_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"Successfully patched {hints_path}")
        else:
            print(f"Already patched or targets not found in {hints_path}")
            
    except Exception as e:
        print(f"Failed to patch torch: {e}")

if __name__ == "__main__":
    patch_torch_inductor()
