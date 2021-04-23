def forward(self, x, c_org, c_trg):

      codes = self.encoder(x, c_org)
       if c_trg is None:
            return torch.cat(codes, dim=-1)

        tmp = []
        for code in codes:
            tmp.append(code.unsqueeze(1).expand(-1, int(x.size(1)/len(codes)), -1))
        code_exp = torch.cat(tmp, dim=1)

        encoder_outputs = torch.cat((code_exp, c_trg.unsqueeze(1).expand(-1, x.size(1), -1)), dim=-1)

        mel_outputs = self.decoder(encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs.transpose(2, 1))
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet.transpose(2, 1)

        mel_outputs = mel_outputs.unsqueeze(1)
        mel_outputs_postnet = mel_outputs_postnet.unsqueeze(1)

        return mel_outputs, mel_outputs_postnet, torch.cat(codes, dim=-1)
