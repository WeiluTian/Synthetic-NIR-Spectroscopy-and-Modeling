classdef positionalEncodingLayer < nnet.layer.Layer
    properties
        SequenceLength
        EmbeddingDim
        PosEnc
    end
    
    methods
        function layer = positionalEncodingLayer(seqLen, embedDim, varargin)
            % Parse optional name-value argument
            name = "";
            if ~isempty(varargin)
                for i = 1:2:length(varargin)
                    if strcmpi(varargin{i}, 'Name')
                        name = varargin{i+1};
                    end
                end
            end
            
            % Call superclass constructor with optional name
            layer.Name = name;
            layer.Description = "Sinusoidal positional encoding";
            
            % Store dimensions
            layer.SequenceLength = seqLen;
            layer.EmbeddingDim = embedDim;
            
            % Create positional encoding matrix
            pos = (0:seqLen-1)';
            i = 0:embedDim-1;
            angleRates = 1 ./ (10000 .^ (2*(floor(i/2))/embedDim));
            angleRads = pos * angleRates;
            PE = zeros(seqLen, embedDim);
            PE(:,1:2:end) = sin(angleRads(:,1:2:end));
            PE(:,2:2:end) = cos(angleRads(:,2:2:end));
            
            % Store as single precision
            layer.PosEnc = single(PE);
        end
        
        function Z = predict(layer, X)
            % X is [embedDim x seqLen x batchSize]
            [embedDim, seqLen, batchSize] = size(X);
        
            if embedDim ~= layer.EmbeddingDim
                error("Embedding dim mismatch: expected %d, got %d", ...
                    layer.EmbeddingDim, embedDim);
            end
        
            pe = layer.PosEnc(1:seqLen,:); % [seqLen x embedDim]
            pe = pe'; % [embedDim x seqLen]
            pe = reshape(pe, [embedDim, seqLen, 1]); % add singleton batch dim
            pe = repmat(pe, [1, 1, batchSize]);
        
            Z = X + pe;
        end

    end
end
