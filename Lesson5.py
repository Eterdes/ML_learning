from enum import Enum
from datetime import datetime
import logging


class AuditLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"


class AuditLog:
    def __init__(self):
        self.logs = []
        logging.basicConfig(filename='audit.log', level=logging.INFO)
        self.logger = logging.getLogger('bank')

    def log(self, level: AuditLevel, message: str):
        # 1. Создаём словарь с логом
        entry = {
            'time': datetime.now().isoformat(),
            'level': level.value,
            'message': message
        }
        # 2. Сохраняем в память
        self.logs.append(entry)
        # 3. Пишем в файл
        self.logger.info(f"{level.value}: {message}")
        # 4. Показываем на экран
        print(f"[{level.value}] {message}")


    def filter(self, min_level: AuditLevel = AuditLevel.INFO):
        level_order = {
            AuditLevel.INFO: 1,
            AuditLevel.WARNING: 2
        }
        result = []
        for log in self.logs:
            if level_order[AuditLevel(log['level'])] >= level_order[min_level]:
                result.append(log)
        return result
    
class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM" 
    HIGH = "HIGH"

class RiskAnalyzer:
    def __init__(self):
        self.audit = AuditLog()  
    
    def analyze(self, amount: float, is_night: bool, to_new_account: bool = False):
        risks = 0
        
        if amount > 100000:
            risks += 1
            self.audit.log(AuditLevel.WARNING, f"🚨 Большая сумма: {amount}")
        
        if is_night:
            risks += 1
            self.audit.log(AuditLevel.WARNING, "🌙 Операция ночью!")
        
        if to_new_account:
            risks += 1
            self.audit.log(AuditLevel.WARNING, "🆕 Новый счёт!")
        
        if risks == 0:
            level = RiskLevel.LOW
        elif risks == 1:
            level = RiskLevel.MEDIUM
        else:
            level = RiskLevel.HIGH
            
        self.audit.log(AuditLevel.INFO, f"Итог: {level.value} ({risks} рисков)")
        return level
    
    def should_block(self, risk: RiskLevel) -> bool:
        if risk == RiskLevel.HIGH:
            self.audit.log(AuditLevel.WARNING, "🚫 БЛОКИРУЕМ операцию!")
            return True
        return False

class AuditReporter:
    def __init__(self, analyzer: RiskAnalyzer):
        self.analyzer = analyzer
    
    def report(self):
        """Отчёт: подозрительные + статистика"""
        all_logs = self.analyzer.audit.logs
        warnings = self.analyzer.audit.filter(AuditLevel.WARNING)
        
        suspicious = [log for log in warnings if "БЛОКИРУЕМ" in log['message']]
        
        print("\n📊 ОТЧЁТ АУДИТА")
        print(f"Всего логов: {len(all_logs)}")
        print(f"Предупреждений: {len(warnings)}")
        print(f"Заблокировано: {len(suspicious)}")
        
        if suspicious:
            print("🚨 Подозрительные:")
            for s in suspicious:
                print(f"  {s['message']}")


print("\n🏦 БАНКОВСКИЕ ТЕСТЫ")
analyzer = RiskAnalyzer()

# Нормальные операции
analyzer.analyze(1000, False, False)   # LOW
analyzer.analyze(50000, False, False)  # LOW/MEDIUM

# ПОДОЗРИТЕЛЬНЫЕ ☠️
analyzer.analyze(200000, True, True)   # HIGH!
analyzer.should_block(RiskLevel.HIGH)

print("\n📈 ОТЧЁТ:")
reporter = AuditReporter(analyzer)
reporter.report()
