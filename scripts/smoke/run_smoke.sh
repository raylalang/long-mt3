for script in $(ls *.py | sort); do
    echo "Running $script..."
    python "$script"
done