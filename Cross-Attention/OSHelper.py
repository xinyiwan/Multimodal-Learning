import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_loss_curves(run_dir: str, save_name: str = "loss_curves.png") -> None:
    """
    Plot training and validation loss curves from the metrics CSV file.
    Creates smooth curves for both train and validation loss.
    
    Args:
        run_dir: Directory containing the training logs (where metrics.csv is located)
        save_name: Filename to save the plot (default: "loss_curves.png")
    """
    metrics_path = os.path.join(run_dir, "metrics.csv")
    
    if not os.path.exists(metrics_path):
        return
    
    try:
        df = pd.read_csv(metrics_path)
    except Exception as e:
        return
    
    # Extract columns related to losses
    train_loss_cols = [col for col in df.columns if "train" in col.lower() and "loss" in col.lower()]
    val_loss_cols = [col for col in df.columns if "val" in col.lower() and "loss" in col.lower()]
    
    if not train_loss_cols and not val_loss_cols:
        return
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot training loss curve
    if train_loss_cols:
        train_col = train_loss_cols[0]
        train_data = df[train_col].dropna()
        ax.plot(
            train_data.index, 
            train_data.values, 
            label="Train Loss", 
            linewidth=2.5, 
            color='#1f77b4',
            linestyle='-',
            alpha=0.8
        )
    
    # Plot validation loss curve
    if val_loss_cols:
        val_col = val_loss_cols[0]
        val_data = df[val_col].dropna()
        ax.plot(
            val_data.index, 
            val_data.values, 
            label="Val Loss", 
            linewidth=2.5, 
            color='#ff7f0e',
            linestyle='-',
            alpha=0.8
        )
    
    # Formatting
    ax.set_xlabel("Epoch", fontsize=13, fontweight='bold')
    ax.set_ylabel("Loss", fontsize=13, fontweight='bold')
    ax.set_title("Training and Validation Loss Curves", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc='best', framealpha=0.95)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=0.7)
    ax.set_xlim(left=0)
    
    # Improve layout
    fig.tight_layout()
    
    # Save the plot
    save_path = os.path.join(run_dir, save_name)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    # Save the plot
    save_path = os.path.join(run_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
