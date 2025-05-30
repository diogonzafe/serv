import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

try:
    from llama_cpp import Llama
    print("✅ llama-cpp-python importado")
    
    # Testar se tem suporte CUDA
    import ctypes
    try:
        cuda = ctypes.CDLL('libcuda.so.1')
        print("✅ CUDA library encontrada")
    except:
        print("❌ CUDA library NÃO encontrada")
    
    # Verificar GPU
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi funciona")
        else:
            print("❌ nvidia-smi não funciona")
    except:
        print("❌ nvidia-smi não encontrado")
        
except Exception as e:
    print(f"❌ Erro: {e}")
