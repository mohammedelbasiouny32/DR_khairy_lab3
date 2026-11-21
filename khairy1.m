%% OFDM system simulation - Project Part 3
% BER vs Eb/N0 for BPSK and 16QAM
% Channels : (1) Rayleigh flat fading, (2) frequency-selective (independent
%                Rayleigh fading per subcarrier, modeled in frequency domain)
% Coding   : none  or  rate-1/5 repetition code (+ zero padding)
% Chain    : bits -> coding -> interleaver -> mapper -> 256-IFFT -> add CP
%            -> channel (+AWGN) -> remove CP -> FFT -> equalize -> demap
%            -> deinterleave -> decode
% Figures  : for each modulation -> 4 single-case figures, then 1 figure
%            with the 4 cases together
clear; clc; close all;

%% ---------------- Parameters ----------------
N        = 256;                 % IFFT size = number of subcarriers
Ncp      = 64;                  % cyclic prefix length (N/4)
nSym     = 2000;                % OFDM symbols per SNR point (increase for smoother curves)
EbN0_dB  = 0:3:30;              % Eb/N0 range in dB
mods     = {'BPSK','16QAM'};
rng(1);

% Case definitions (order: s = 1..4)
chanList  = {'flat','flat','selective','selective'};
codedList = [0 1 0 1];
labels = {'Flat fading - no coding', 'Flat fading - repetition 1/5', ...
          'Freq-selective - no coding', 'Freq-selective - repetition 1/5'};
colors = [0.00 0.45 0.74;      % blue
          0.85 0.33 0.10;      % orange-red
          0.13 0.60 0.30;      % green
          0.55 0.15 0.60];     % purple
lstyle = {'-','--','-','--'};
marker = {'o','s','^','d'};

for m = 1:numel(mods)
    mod = mods{m};
    switch mod
        case 'BPSK',  bps = 1; R = 16; C = 16;   % 16x16 = 256 bits = 1 BPSK OFDM symbol
        case '16QAM', bps = 4; R = 32; C = 32;   % 32x32 = 1024 bits
    end
    Nb = N*bps;                 % coded bits that fill ONE OFDM symbol
    nO = R*C/Nb;                % OFDM symbols covered by ONE interleaver block (1 for both BPSK and 16QAM here)
    nBlk = ceil(nSym/nO);       % number of interleaver blocks per SNR point
    BER = cell(1,4);

    %% ---- Simulation of the 4 cases ----
    for s = 1:4
        chan = chanList{s};  coded = codedList(s);
        if coded, Kd = floor(Nb/5); else, Kd = Nb; end   % info bits per OFDM symbol
        ber = zeros(size(EbN0_dB));

        for i = 1:numel(EbN0_dB)
            Eb = N / Kd;                         % energy per INFORMATION bit
            N0 = Eb / 10^(EbN0_dB(i)/10);
            nErr = 0;

            for k = 1:nBlk
                % ---------- Transmitter ----------
                data = randi([0 1], Kd, nO);               % one column per OFDM symbol
                if coded
                    cb = repelem(data, 5, 1);              % rate 1/5 repetition
                    cb = [cb; zeros(Nb-5*Kd, nO)];         % zero padding to Nb bits per OFDM symbol
                else
                    cb = data;
                end
                ib = reshape(interleave(cb(:), R, C), Nb, nO);   % interleave the whole R*C block
                rb = zeros(Nb, nO);

                for q = 1:nO                               % each OFDM symbol of the block
                    X   = mapper(ib(:,q), mod);            % N x 1 symbols
                    x   = sqrt(N)*ifft(X);                 % unit average power per sample
                    xcp = [x(end-Ncp+1:end); x];           % add cyclic prefix

                    % ---------- Channel ----------
                    if strcmp(chan,'flat')
                        H = (randn+1j*randn)/sqrt(2) * ones(N,1);  % one Rayleigh gain for all subcarriers
                    else
                        H = (randn(N,1)+1j*randn(N,1))/sqrt(2);    % independent Rayleigh per subcarrier
                    end
                    tmp  = xcp(Ncp+1:end);                 % fading applied in frequency domain
                    tmp  = ifft(fft(tmp).*H);
                    rxcp = [tmp(end-Ncp+1:end); tmp];
                    rxcp = rxcp + sqrt(N0/2)*(randn(size(rxcp))+1j*randn(size(rxcp)));  % AWGN

                    % ---------- Receiver ----------
                    r   = rxcp(Ncp+1:end);                 % remove CP
                    Y   = fft(r)/sqrt(N);
                    Yeq = Y ./ H;                          % zero-forcing equalizer (perfect CSI)
                    rb(:,q) = demapper(Yeq, mod);
                end

                db = reshape(deinterleave(rb(:), R, C), Nb, nO);
                if coded
                    dec = zeros(Kd, nO);
                    for q = 1:nO
                        blk = reshape(db(1:5*Kd,q), 5, Kd);
                        dec(:,q) = (sum(blk,1) >= 3).';    % majority vote
                    end
                else
                    dec = db;
                end
                nErr = nErr + sum(dec(:) ~= data(:));
            end
            ber(i) = nErr/(nBlk*nO*Kd);
        end
        BER{s} = ber;
    end

    g = 10.^(EbN0_dB/10);
    theory = 0.5*(1-sqrt(g./(1+g)));       % uncoded BPSK, flat Rayleigh

    %% ---- Figures 1-4: each case ALONE ----
    for s = 1:4
        figure('Color','w','Position',[100 80 900 620]); hold on;
        b = BER{s}; b(b==0) = NaN;
        semilogy(EbN0_dB, b, 'Color',colors(s,:), 'LineStyle',lstyle{s}, ...
                 'Marker',marker{s}, 'MarkerSize',9, 'LineWidth',2.5, ...
                 'MarkerFaceColor',colors(s,:), 'DisplayName',labels{s});
        if strcmp(mod,'BPSK') && s == 1
            semilogy(EbN0_dB, theory, 'k:', 'LineWidth',2.5, ...
                     'DisplayName','Theory: uncoded flat BPSK');
        end
        styleAxes([mod ' - ' labels{s}], EbN0_dB);
        saveas(gcf, sprintf('OFDM_%s_case%d.png', mod, s));
    end

    %% ---- Figure 5: all 4 cases TOGETHER ----
    figure('Color','w','Position',[100 80 950 650]); hold on;
    for s = 1:4
        b = BER{s}; b(b==0) = NaN;
        semilogy(EbN0_dB, b, 'Color',colors(s,:), 'LineStyle',lstyle{s}, ...
                 'Marker',marker{s}, 'MarkerSize',9, 'LineWidth',2.5, ...
                 'MarkerFaceColor',colors(s,:), 'DisplayName',labels{s});
    end
    if strcmp(mod,'BPSK')
        semilogy(EbN0_dB, theory, 'k:', 'LineWidth',2.5, ...
                 'DisplayName','Theory: uncoded flat BPSK');
    end
    styleAxes([mod ' - all cases'], EbN0_dB);
    saveas(gcf, ['OFDM_' mod '_all.png']);
end

%% ---------------- Local functions ----------------
function styleAxes(ttl, EbN0_dB)
    set(gca,'YScale','log','FontSize',13,'LineWidth',1.2,'Box','on', ...
            'GridAlpha',0.35,'MinorGridAlpha',0.15);
    grid on; grid minor;
    xlabel('E_b/N_0 (dB)','FontSize',15,'FontWeight','bold');
    ylabel('Bit Error Rate (BER)','FontSize',15,'FontWeight','bold');
    title(['OFDM (256 subcarriers) - ' ttl],'FontSize',16);
    legend('Location','southwest','FontSize',12);
    xlim([EbN0_dB(1) EbN0_dB(end)]); ylim([1e-5 1]);
end

function out = interleave(bits, R, C)      % write row-wise, read column-wise
    M = reshape(bits, C, R).';
    out = M(:);
end

function out = deinterleave(bits, R, C)    % inverse operation
    M = reshape(bits, R, C);
    out = reshape(M.', [], 1);
end

function s = mapper(b, mod)
    switch mod
        case 'BPSK'
            s = 2*b - 1;                                   % 0 -> -1, 1 -> +1
        case '16QAM'                                       % Gray: 00->-3, 01->-1, 11->+1, 10->+3
            B = reshape(b,4,[]);
            I = (2*B(1,:)-1).*(3-2*B(2,:));
            Q = (2*B(3,:)-1).*(3-2*B(4,:));
            s = (I + 1j*Q).'/sqrt(10);                     % unit average energy
    end
end

function b = demapper(s, mod)
    switch mod
        case 'BPSK'
            b = real(s) > 0;
        case '16QAM'
            I = real(s).'*sqrt(10);  Q = imag(s).'*sqrt(10);
            B = [I > 0; abs(I) < 2; Q > 0; abs(Q) < 2];
            b = B(:);
    end
    b = double(b);
end