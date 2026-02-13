% 修改 simulink.m 文件 - 使用正确的网格格式
load('P_interp_data.mat');  % 加载 x_grid 和 P

% 检查数据大小
fprintf('原始数据大小: x_grid = %s, P = %s\n', mat2str(size(x_grid)), mat2str(size(P)));

% 使用下采样 - 减少数据量
target_points = 5000; % 目标数据点数
if size(x_grid, 1) > target_points
    downsample_factor = ceil(size(x_grid, 1) / target_points);
    indices = 1:downsample_factor:size(x_grid, 1);
    x_grid_down = x_grid(indices, :);
    P_down = P(indices, :, :);
else
    x_grid_down = x_grid;
    P_down = P;
end

fprintf('下采样后数据大小: x_grid = %s, P = %s\n', mat2str(size(x_grid_down)), mat2str(size(P_down)));

% 创建网格数据 - 使用 ndgrid 而不是 meshgrid
x1_unique = unique(double(x_grid_down(:,1)));
x2_unique = unique(double(x_grid_down(:,2)));

% 进一步减少网格点数以避免内存问题
max_grid_points = 100; % 限制最大网格点数
if length(x1_unique) > max_grid_points
    x1_unique = linspace(min(x1_unique), max(x1_unique), max_grid_points);
end
if length(x2_unique) > max_grid_points
    x2_unique = linspace(min(x2_unique), max(x2_unique), max_grid_points);
end

fprintf('网格大小: x1 = %d, x2 = %d\n', length(x1_unique), length(x2_unique));

% 使用 ndgrid 创建网格（griddedInterpolant 需要的格式）
[X1, X2] = ndgrid(x1_unique, x2_unique);

% 使用 scatteredInterpolant 将散乱数据插值到规则网格
F11_scattered = scatteredInterpolant(double(x_grid_down(:,1)), double(x_grid_down(:,2)), double(squeeze(P_down(:,1,1))));
F12_scattered = scatteredInterpolant(double(x_grid_down(:,1)), double(x_grid_down(:,2)), double(squeeze(P_down(:,1,2))));
F22_scattered = scatteredInterpolant(double(x_grid_down(:,1)), double(x_grid_down(:,2)), double(squeeze(P_down(:,2,2))));

% 在规则网格上评估 scatteredInterpolant
F11_values = F11_scattered(X1, X2);
F12_values = F12_scattered(X1, X2);
F22_values = F22_scattered(X1, X2);

% 创建 griddedInterpolant（使用 ndgrid 格式）
F11_grid = griddedInterpolant(X1, X2, F11_values);
F12_grid = griddedInterpolant(X1, X2, F12_values);
F22_grid = griddedInterpolant(X1, X2, F22_values);

% 保存网格插值器
save('P_gridded_interpolants.mat', 'F11_grid', 'F12_grid', 'F22_grid');

fprintf('网格插值数据已保存，使用 ndgrid 格式\n');


fc = 10;          % 截止频率 10 Hz（先用这个）
wc = 2*pi*fc;     % 角频率
