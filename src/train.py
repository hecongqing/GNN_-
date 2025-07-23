import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import argparse
import json

from .model import LSTMSentimentModel
from .dataset import SentimentDataProcessor
from .utils import setup_logger, save_model, load_model, EarlyStopping


class SentimentTrainer:
    """LSTM情感分析训练器"""
    
    def __init__(self, model, device=None, logger=None):
        """
        初始化训练器
        
        Args:
            model: LSTM模型
            device: 计算设备
            logger: 日志记录器
        """
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logger if logger else setup_logger('SentimentTrainer')
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 初始化TensorBoard
        self.writer = None
        
        # 训练状态
        self.global_step = 0
        self.best_val_acc = 0.0
        
    def train_epoch(self, train_loader, optimizer, epoch):
        """
        训练一个epoch
        
        Args:
            train_loader: 训练数据加载器
            optimizer: 优化器
            epoch: 当前epoch
            
        Returns:
            平均损失和准确率
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            lengths = batch['length'].to(self.device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = self.model(input_ids, lengths)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 统计
            total_loss += loss.item()
            predicted = torch.argmax(outputs, dim=1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            acc = total_correct / total_samples
            progress_bar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{acc:.4f}'
            })
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Loss', loss.item(), self.global_step)
                self.writer.add_scalar('Train/Accuracy', acc, self.global_step)
            
            self.global_step += 1
        
        avg_loss = total_loss / len(train_loader)
        avg_acc = total_correct / total_samples
        
        return avg_loss, avg_acc
    
    def validate(self, val_loader):
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
            
        Returns:
            验证损失、准确率、精确率、召回率、F1分数
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predicted = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001,
              output_dir='outputs', save_best=True, patience=5):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            learning_rate: 学习率
            output_dir: 输出目录
            save_best: 是否保存最佳模型
            patience: 早停耐心值
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        
        # 初始化TensorBoard
        log_dir = os.path.join(output_dir, 'logs', datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.writer = SummaryWriter(log_dir)
        
        # 优化器和学习率调度器
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=2, verbose=True
        )
        
        # 早停
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        
        self.logger.info(f"开始训练，共 {num_epochs} 个epoch")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"学习率: {learning_rate}")
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, epoch)
            
            # 验证
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.validate(val_loader)
            
            # 学习率调度
            scheduler.step(val_acc)
            
            epoch_time = time.time() - start_time
            
            # 记录日志
            self.logger.info(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Val F1: {val_f1:.4f} - Time: {epoch_time:.2f}s"
            )
            
            # 记录到TensorBoard
            if self.writer:
                self.writer.add_scalar('Validation/Loss', val_loss, epoch)
                self.writer.add_scalar('Validation/Accuracy', val_acc, epoch)
                self.writer.add_scalar('Validation/F1', val_f1, epoch)
                self.writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 保存最佳模型
            if save_best and val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                model_path = os.path.join(output_dir, 'model_best.pt')
                save_model(self.model, model_path, epoch, val_acc)
                self.logger.info(f"保存最佳模型到 {model_path}")
            
            # 早停检查
            early_stopping(val_loss)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break
        
        # 关闭TensorBoard
        if self.writer:
            self.writer.close()
        
        self.logger.info("训练完成！")
    
    def test(self, test_loader, model_path=None):
        """
        测试模型
        
        Args:
            test_loader: 测试数据加载器
            model_path: 模型路径
            
        Returns:
            测试结果
        """
        if model_path:
            load_model(self.model, model_path)
            self.logger.info(f"加载模型: {model_path}")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Testing'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                lengths = batch['length'].to(self.device)
                
                outputs = self.model(input_ids, lengths)
                predicted = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        
        # 详细报告
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        self.logger.info("测试结果:")
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Precision: {precision:.4f}")
        self.logger.info(f"Recall: {recall:.4f}")
        self.logger.info(f"F1 Score: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'report': report
        }


def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='LSTM情感分析训练')
    parser.add_argument('--data_dir', type=str, default='dataset', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='outputs', help='输出目录')
    parser.add_argument('--max_vocab_size', type=int, default=10000, help='最大词汇表大小')
    parser.add_argument('--max_length', type=int, default=128, help='最大序列长度')
    parser.add_argument('--embedding_dim', type=int, default=128, help='词嵌入维度')
    parser.add_argument('--hidden_dim', type=int, default=256, help='LSTM隐藏层维度')
    parser.add_argument('--num_layers', type=int, default=2, help='LSTM层数')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout率')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--patience', type=int, default=5, help='早停耐心值')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 初始化数据处理器
    data_processor = SentimentDataProcessor(
        data_dir=args.data_dir,
        max_vocab_size=args.max_vocab_size,
        max_length=args.max_length
    )
    
    # 准备数据
    train_loader, val_loader, test_loader = data_processor.prepare_data()
    
    # 保存词汇表和标签编码器
    os.makedirs(args.output_dir, exist_ok=True)
    data_processor.save_vocab(os.path.join(args.output_dir, 'vocab.pkl'))
    data_processor.save_label_encoder(os.path.join(args.output_dir, 'label_encoder.pkl'))
    
    # 初始化模型
    model = LSTMSentimentModel(
        vocab_size=data_processor.get_vocab_size(),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=data_processor.get_num_classes(),
        dropout=args.dropout
    )
    
    # 初始化权重
    model.init_weights()
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 初始化训练器
    trainer = SentimentTrainer(model, device)
    
    # 训练模型
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        patience=args.patience
    )
    
    # 测试模型
    best_model_path = os.path.join(args.output_dir, 'model_best.pt')
    test_results = trainer.test(test_loader, best_model_path)
    
    # 保存测试结果
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w', encoding='utf-8') as f:
        json.dump(test_results, f, ensure_ascii=False, indent=2)
    
    print("训练和测试完成！")


if __name__ == "__main__":
    main()