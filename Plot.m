figure;

semilogy(SNR_Range,MSE, 'color', [0 0.4470 0.7410], 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_10, 'color', [0.8500 0.3250 0.0980], 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_20, 'color', [0.9290 0.6940 0.1250], 'LineWidth', 1);
hold on
semilogy(SNR_Range,MSE_30, 'color', [0.3010 0.7450 0.9330], 'LineWidth', 1);

ylim([0.0001, 1])

legend('InterpolationResNet-10F', ...
    'InterpolationResNet-10F 10% pruning', ...
    'InterpolationResNet-10F 20% pruning', ...
    'InterpolationResNet-10F 30% pruning');
xlabel('SNR in dB');
ylabel('MSE');
grid on;
hold off;
